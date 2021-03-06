"""Various helper functions for xmen list"""
import re
from xmen.server import *

DEFAULTS = {
    'display_date': "_timestamps|_created",
    'display_purpose': "_purpose",
    "display_version": "_version",
    'display_status': r'_status$',
    'display_messages': '_messages_(last$|e$|s$|wall$|end$|next$|.*step$|.*load$)',
    'display_meta': '_meta_(root$|name$|mac$|host$|user$|home$)'
}


def args_to_filters(args):
    """Convert args namespace object to filters."""
    # if len(args.filter_params) == 0:
    #     args.filter_params = [".*"]
    filters = [
        '_root',
        '_name',
        args.display_date,
        args.display_status,
        args.display_purpose,
        args.display_messages,
        args.display_version,
        args.display_meta,
        *args.filters]
        # *args.filter_params]
    filters = [split_operators(f) for f in filters]
    # n = len(filters) - len(args.filter_params)
    # filters = [[(r'(?!_)(' if i > n - 1 else '(') + f[0] + ')', *f[1:]] for i, f in enumerate(filters)]
    return filters


def split_operators(m):
    """Convert a string of the form 'regex op y' or 'regex' to a list of parts. Useful for generating
    filters for the visualise_params function."""
    splits = re.split(r'(<=|==|>=|<|>|!=)', m.replace(' ', ''))
    if len(splits) in [1, 3]:
        return splits
    else:
        raise RuntimeError(f'invalid parameter config passed {m}')


def visualise_params(dics, *filters, roots=None, short_root=False):
    """Visualise a set of experiment parameters as a data-frame.

    Args:
        dics: A list of dictionaries loaded from param.yml files. Do not need to contain the same keys
        roots: optional names for each element in dics
        *filters: If filter is of length 1 it is treated as a regex and keys from each experiment matching the regex
            will be displayed in the data_frame. If filter is a triplet ``reg``, ``op``, ``y``,
            then experiments (rows) will only be added to the table if __all__ the keys matching
            reg satisfy the condition exec({k} {op} {y}). If more than one filter is passed the result will be the
            combination of each of the filters applied individually.

    Returns:
        df: A pandas data frame of the experiments matching the filters. Rows are experiments and columns are
            parameters.
    """
    import numpy as np
    from xmen.utils import dics_to_pandas
    import os
    # convert parameters to pandas data-frame
    df = dics_to_pandas(dics, '|'.join(f[0] for f in filters))
    keys = df.columns
    # remove '_name'
    if '_name' in df:
        names = df.pop('_name')
        df['_root'] += os.path.sep + names
    if roots is not None:
        df['_root'] = roots
    # filter parameters if filters are a tuple of (reg, op, y)
    for k in keys:
        for f in filters:
            if len(f) == 3 and re.match(f[0], k):
                reg, op, y = f
                _ = {}
                exec(f'df = df.loc[df[k] {op} {y}]', {'df': df, 'k': k}, _)
                df = _['df']

    # fix table order
    display_keys = ['_root', '_status', '_user', '_host', '_purpose']
    display_keys += [k for k in keys if not k.startswith('_')]
    display_keys += [k for k in keys if k.startswith('_timestamps')]
    display_keys += [k for k in keys if k.startswith('_messages')]
    display_keys += [k for k in keys if k.startswith('_meta')]
    display_keys += [k for k in keys if k not in display_keys]
    df = df.filter(items=display_keys)
    # update the names of the table
    from collections import OrderedDict
    names = [
        ('_root', 'root'), ('_name', 'name'), ('_status', 'status'), ('_purpose', 'purpose'), ('_host', 'host'),
        ('_user', 'user'),
        ('_created', 'created'), ('_messages_', ''), ('_version_', ''),
        ('_meta_', ''), ('_timestamps_', ''), ('_meta_slurm_', 's')]
    updates = {}
    for k in keys:
        for s, n in names:
            if k.startswith(s):
                updates[k] = k.replace(s, n)
    df = df.rename(updates, axis='columns')
    df = df.replace(np.nan, '', regex=True)

    max_ch = 70
    if short_root:
        _ = []
        for r in df['root']:
            r = r.split(':')
            if len(r) > 2:
                r = [r[0], ':'.join(r[1:])]
            r[0] = r[0] + ':'
            if len(r[0]) + len(r[1]) > max_ch:
                remainder = r[1].split(os.sep)
                back = []
                for rr in reversed(remainder):
                    back += [rr]
                    if sum(len(x) for x in back) > max_ch - len(r[0]):
                        break
                if len(back) > 1:
                    back = back[:-1]
                r[1] = ' ... ' + os.sep.join(reversed(back))
            _ += [''.join([r[0], r[1]])]
        df['root'] = _

    # shorten paths for visualisation
    import os

    prefix = ''
    # roots = [v["root"] for v in df.transpose().to_dict().values()]
    # hosts = set(v.split('@')[1] for v in roots if len(v.split('@')) > 1)
    # if len(hosts) == 1:
    #     roots = [v["root"] for v in df.transpose().to_dict().values()]
    #     prefix = os.path.dirname(os.path.commonprefix(roots))
    #     if prefix != '/':
    #         roots = [r[len(prefix) + 1:] for r in roots]
    #     df.update({'root': roots})
    # else:
    #     prefix = ''

    # get_rid = {
    #     'sid',
    #     'sMCS_label',
    #     'sNice'
    #     'sQOS', 'sTimeMin', 'sEligibleTime', 'sAccrueTime', 'sDeadline',
    #            'sSecsPreSuspend', 'sJobName', 'sSuspendTime', 'sLastSchedEval', 'sReqNodeList', 'sExcNodeList',
    #            'sNice', }
    # filter slurm keys
    skeys = [k.replace('_meta_slurm_', 's') for k in keys if k.startswith('_meta_slurm_')]
    keep = {'JobId', 'UserId', 'GroupId', 'Priority', 'Account', 'TimeLimit',
            'StartTime', 'EndTime', 'NodeList', 'NumNodes',
            'NumCPUSs', 'MailUser', 'MailType'}
    get_rid = [k for k in skeys if k[1:] not in keep]
    df = df.filter(items=[k for k in df.columns if k not in get_rid])
    return df, prefix


def notebook_display(results, *filters):
    """Display experiments in notebook mode"""
    from xmen.utils import dics_to_pandas
    import os

    results = dics_to_pandas(results, '|'.join(f[0] for f in filters))
    sets = [os.path.split(r) for r in results['_root'].to_list()]

    notebook = {}
    for i, (s, name) in enumerate(sets):
        if s not in notebook:
            notebook[s] = []
        data = results.T[i].to_dict()
        data['_name'] = name
        notebook[s] += [data]

    note = ['']
    k = 2
    for i, (s, v) in enumerate(notebook.items()):
        note += [str(i) + ':' + s]
        purpose, notes, last, git = [], [], [], []
        for vv in v:
            note += [(' ' * k) + '|- ' + vv['_name']]
            purpose += [vv['_purpose']]
            n = vv.get('_notes', None)
            if n is None:
                n = []
            notes.extend(n)
            ts = vv.get('_timestamps_last', None)
            if ts is None:
                ts = vv.get('_timestamps_registered', None)
            if ts is None:
                ts = vv.get('_created', '')
            last += [ts]

            g = vv.get('_version_git_local', '')
            g += ' ' + vv.get('_version_git_branch', '')
            g += ' ' + vv.get('_version_git_commit', '')
            if g != '  ':
                git += [g]

        purpose, notes, git = set(purpose), set(notes), set(git)
        note += [f'Last: {max(last)}']
        note += [f'Git:']
        for g in git:
            note.extend([(' ' * k) + gg for gg in g.split(' ')])
        note += ['Purpose:']
        if purpose:
            note += [(' ' * k) + p for p in purpose]
        note += ['Notes:']
        if notes:
            note += [(' ' * k) + p for p in set(notes)]
        note += ['']
    print('\n'.join(note))


class NotEnoughRows(Exception):
    pass


def manage_backspace(x):
    """A validator function to manage backspacing in curses textbox. For some reason my backspace maps to 127
    (key delete)"""
    import curses
    if x in [127]:
        x = curses.KEY_BACKSPACE
    return x


def extract_results(response):
    from xmen.utils import dic_from_yml
    from collections import OrderedDict
    updates = OrderedDict()
    if response['matches']:
        uid, root, status, user, added, updated, data = zip(*response['matches'])
        data = [dic_from_yml(string=d) for d in data]
        for d, s in zip(data, status):
            d['_status'] = s
        updates = OrderedDict([
            (i, [r, s, u, str(a), str(up), d]) for i, r, s, u, a, up, d
            in zip(uid, root, status, user, added, updated, data)])
    return updates, response['time']


def process_results_task(q_response, q_results):
    while True:
        response = q_response.get()
        updates, time = extract_results(response)
        q_results.put((updates, time))


def update_requests(requests_q, last_request_time, config):
    import datetime
    request_from = datetime.datetime.strptime(last_request_time, '%Y-%m-%d %H:%M:%S')
    # allow grace of 10 seconds
    request_from -= datetime.timedelta(seconds=10.)
    request = GetExperiments(
        config.user,
        config.password,
        '.*',
        '.*',
        request_from.strftime('%Y-%m-%d %H:%M:%S'),
        None)
    requests_q.put(request, block=True)


def interactive_display(stdscr, args):
    """interactively display results with various search queries"""
    import curses
    import multiprocessing as mp
    import queue
    from xmen.utils import dic_from_yml
    from collections import OrderedDict

    global root
    global results
    global default_pattern
    global expand_helps
    global short_root
    global last_request_time
    default_pattern = None
    expand_helps = False
    short_root = True
    stdscr.refresh()
    rows, cols = stdscr.getmaxyx()

    if rows < 9:
        raise NotEnoughRows

    pos_x, pos_y = 0, 0

    # load cached results
    from xmen.config import Config
    config = Config()
    results, last_request_time = config.cache(load=True)
    meta, message = [], []

    # initialise colours
    if curses.has_colors():
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    WHITE = curses.color_pair(1)
    RED = curses.color_pair(2)
    CYAN = curses.color_pair(3)
    YELLOW = curses.color_pair(4)
    GREEN = curses.color_pair(5)
    MAGNETA = curses.color_pair(6)

    global window
    window = curses.newwin(rows, cols, 0, 0)

    def generate_table(results, args):
        dics, roots = results.values(), results.keys()
        data_frame, root = visualise_params(dics, *args_to_filters(args), roots=roots, short_root=short_root)
        return data_frame, root

    def toggle(args, name, update=None):
        global default_pattern
        val = getattr(args, name)
        if name == 'display_meta':
            if update in meta:
                meta.remove(update)
            else:
                meta.append(update)
            update = '|'.join(meta)
            setattr(args, name, update if update else '^$')
        elif name == 'display_messages':
            if update in message:
                message.remove(update)
            else:
                message.append(update)
            update = '|'.join(message)
            setattr(args, name, update if update else '^$')
        elif name == 'filters':
            if update == '':
                setattr(args, name, [args.filters[0]])
            else:
                setattr(args, name, getattr(args, name) + [update])
        elif name == 'filter_params':
            if update == '':
                setattr(args, name, [args.filter_params[0]])
            else:
                setattr(args, name, getattr(args, name) + [update])
        elif name == 'status_filter':
            if update == '':
                setattr(args, name, '[^(deleted)]')
            else:
                setattr(args, name, update)
        elif name == 'pattern':
            if default_pattern is None:
                default_patten = args.pattern

            if update == '':
                setattr(args, name, default_patten)
            else:
                setattr(args, name, update)
        else:
            if update is None:
                update = DEFAULTS[name]
            if val == '^$':
                setattr(args, name, update)
            else:
                setattr(args, name, '^$')
        return args

    def display_row(i, pad, table=None, x=None):
        from tabulate import tabulate
        if x is None:
            x = tabulate(table, headers='keys', tablefmt='github').split('\n')
        x0 = x[0]
        xi = x[i]
        # stdscr.addstr(i + 2, 1, xx)
        root = [ii for ii, hh in enumerate(x0.split('|')) if hh.strip() == 'root']
        status = [ii for ii, hh in enumerate(x0.split('|')) if hh.strip() == 'status']
        if i == 0:
            xi = xi.replace('|', '')
            pad.addstr(i, 1, xi, curses.A_BOLD)
        else:
            componenents = xi.split('|')
            if status:
                for j, c in enumerate(componenents):
                    if j == status[0]:
                        col = {
                            'deleted': curses.A_DIM,
                            'registered': WHITE, 'timeout': CYAN,
                            'stopped': MAGNETA,
                            'running': YELLOW, 'error': RED,
                            'finished': GREEN}
                        cc = col.get(c.strip(), None)
                        if cc is None:
                            offset = sum(len(_) for _ in componenents[:j]) + 1
                            pad.addstr(i, offset, c)
                        else:
                            pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c, cc)
                    else:
                        pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c)
            else:
                pad.addstr(i, 1, ''.join(componenents))

    def display_table(table):
        import curses
        global pad, window, expand_helps
        rows, cols = stdscr.getmaxyx()
        window.erase()

        if last_time is not None:
            window.addstr(0, 0, f'Last update recieved @ {last_request_time}', curses.A_DIM)
        else:
            window.addstr(0, 0, f'No running experiments', curses.A_DIM)

        legend_pad = curses.newpad(5, 500)
        if expand_helps:
            legend_pad.addstr(0, 0, 'Help: (press o to minimise)', curses.A_BOLD)
            legend_pad.addstr(1, 0, 'R=expand-roots d=date s=status v=version p=purpose t=monitor-message M=meta ')
            legend_pad.addstr(2, 0, 'G=gpu S=slurm c=cpu n=network V=virtual w=swap O=os D=disks')
            legend_pad.addstr(3, 0, 'r=filter-roots   z = filter-status   f=filter-parameters')
            legend_pad.addstr(4, 0, '(press r, z, f for more info)')
        else:
            legend_pad.addstr(0, 0, 'Toggles:', curses.A_BOLD)
            legend_pad.addstr(1, 0,
                              f'date = {args.display_date} meta = {args.display_meta}, messages={args.display_messages}, version={args.display_version}')
            legend_pad.addstr(2, 0, f'pattern = {args.pattern}, status = {args.status_filter}')
            legend_pad.addstr(3, 0, f'filters={args.filters}')
            legend_pad.addstr(4, 0, f'(for more help press o)')

        if table is not None:
            from tabulate import tabulate
            x = tabulate(table, headers='keys', tablefmt='github').split('\n')
            x.pop(1)

            if len(x) > rows - 6:
                filler = '|'.join([' ...'.ljust(len(xx), ' ') if xx else ''
                                     for ii, xx in enumerate(x[1].split('|'))])
                filler = filler.replace('...', '   ', 1)
                x = [x[0], filler] + x[-(rows - 9):]

            pad = curses.newpad(len(x), len(x[0]))
            for i, xx in enumerate(x):
                display_row(i, pad, x=x)
        else:
            pad = curses.newpad(3, 1000)
            pad.addstr(0, 0, '')
            pad.addstr(1, 4, f'No experiments found!', RED | curses.A_BOLD)
            pad.addstr(2, 4, f'Try resetting the filters...')

        window.noutrefresh()
        pad.noutrefresh(pos_x, 0, 1, 0, rows - 5, cols - 1)
        legend_pad.noutrefresh(0, 0, rows - 5, 0, rows - 1, cols - 1)
        curses.doupdate()

    def visualise_results(results, args):
        dics = OrderedDict([
            (r, d) for k, (r, s, u, a, up, d) in results.items()
            if re.match(args.pattern, r)
            and re.match(args.status_filter, s)])

        data_frame = None
        if dics:
            data_frame, root = generate_table(dics, args)
        display_table(data_frame)


    try:
        from xmen.server import send_request_task
        from xmen.utils import dic_from_yml
        import time
        import multiprocessing

        manager = mp.Manager()
        q_request = manager.Queue(maxsize=1)
        q_response = manager.Queue(maxsize=1)
        q_processed = manager.Queue(maxsize=1)

        update_requests(q_request, last_request_time, config)

        p_request = multiprocessing.Process(target=send_request_task, args=(q_request, q_response))
        p_request.start()
        p_process = multiprocessing.Process(target=process_results_task, args=(q_response, q_processed))
        p_process.start()

        # get initial results view
        if not results:
            stdscr.addstr(0, 1, 'Downloading user content from server. This could take a minute but only needs to happen once...', curses.A_BOLD)
            stdscr.refresh()
            try:
                updates, last_request_time = q_processed.get(timeout=180.)
            except queue.Empty:
                stdscr.addstr(0, 1, 'Timeout occurred after 180s. Is the server connected?', curses.A_BOLD)
                return
            results.update(updates)
            config.cache(save=(results, last_request_time))
            update_requests(q_request, last_request_time, config)

        last_time = time.time()
        visualise_results(results, args)
        while True:
            if time.time() - last_time > args.interval:
                # request experiment updates
                try:
                    updates, last_request_time = q_processed.get(False)
                except queue.Empty:
                    pass
                else:
                    results.update(updates)
                    config.cache(save=(results, last_request_time))
                    visualise_results(results, args)
                    update_requests(q_request, last_request_time, config)
                    last_time = time.time()

            if stdscr is not None:
                stdscr.timeout(1000)
                c = stdscr.getch()
                if c == ord('d'):
                    toggle(args, 'display_date')
                    # args.display_date = not args.display_date
                    visualise_results(results, args)
                if c == ord('v'):
                    toggle(args, 'display_version')
                    visualise_results(results, args)
                if c == ord('s'):
                    toggle(args, 'display_status')
                    visualise_results(results, args)
                if c == ord('p'):
                    toggle(args, 'display_purpose')
                    visualise_results(results, args)
                if c == ord('M'):
                    toggle(args, 'display_meta', '_meta_(slurm_job|root$|name$|mac$|host$|user$|home$)')
                    visualise_results(results, args)
                if c == ord('V'):
                    toggle(args, 'display_meta', '_meta_virtual.*')
                    visualise_results(results, args)
                if c == ord('w'):
                    toggle(args, 'display_meta', '_meta_swap.*')
                    visualise_results(results, args)
                if c == ord('O'):
                    toggle(args, 'display_meta', '_meta_system.*')
                    visualise_results(results, args)
                if c == ord('D'):
                    toggle(args, 'display_meta', '_meta_disks.*')
                    visualise_results(results, args)
                if c == ord('c'):
                    toggle(args, 'display_meta', '_meta_cpu.*')
                    visualise_results(results, args)
                if c == ord('G'):
                    toggle(args, 'display_meta', '_meta_gpu.*')
                    visualise_results(results, args)
                if c == ord('n'):
                    toggle(args, 'display_meta', '_meta_network.*')
                    visualise_results(results, args)
                if c == ord('S'):
                    args = toggle(args, 'display_meta', '_meta_slurm.*')
                    visualise_results(results, args)
                if c == ord('t'):
                    args = toggle(args, 'display_messages', '_messages_(last|e$|s$|wall$|end$|next$|m_step$|m_load$)')
                    visualise_results(results, args)
                if c == ord('f'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    window.move(rows - 5, 0)
                    window.clrtoeol()
                    window.move(rows - 4, 0)
                    window.clrtoeol()
                    window.move(rows - 3, 0)
                    window.clrtoeol()
                    window.move(rows - 2, 0)
                    window.clrtoeol()
                    window.move(rows - 1, 0)
                    window.clrtoeol()
                    window.addstr(rows - 3, 0,
                                  'Search: (enter "" to reset)',
                                  curses.A_BOLD)
                    window.addstr(rows-2, 0, 'eg. none to reset, ".*", "_messages_.*", "_meta_.*", _version_git_branch=="xmen", a==10, loss<2.0')
                    window.refresh()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    text_box = Textbox(win)
                    text = text_box.edit(validate=manage_backspace).strip()
                    if text and not text.startswith('_'):
                        text = '(?!_)' + text
                    args = toggle(args, 'filters', text)
                    visualise_results(results, args)
                if c == ord('R'):
                    short_root = not short_root
                    visualise_results(results, args)
                if c == ord('r'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    window.move(rows - 5, 0)
                    window.clrtoeol()
                    window.move(rows - 4, 0)
                    window.clrtoeol()
                    window.move(rows - 3, 0)
                    window.clrtoeol()
                    window.move(rows - 2, 0)
                    window.clrtoeol()
                    window.move(rows - 1, 0)
                    window.clrtoeol()
                    window.addstr(rows - 3, 0,
                                  'Root: (enter "" to reset)',
                                  curses.A_BOLD)
                    window.addstr(rows - 2, 0,
                                  'eg. ".*", "{user}@{host}:{pattern}" where user, host and pattern are regex')
                    window.refresh()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    text_box = Textbox(win)
                    text = text_box.edit(validate=manage_backspace).strip()
                    toggle(args, 'pattern', text.strip())
                    visualise_results(results, args)
                if c == ord('z'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    window.move(rows - 5, 0)
                    window.clrtoeol()
                    window.move(rows - 4, 0)
                    window.clrtoeol()
                    window.move(rows - 3, 0)
                    window.clrtoeol()
                    window.move(rows - 2, 0)
                    window.clrtoeol()
                    window.move(rows - 1, 0)
                    window.clrtoeol()
                    window.addstr(rows - 3, 0,
                                  'Status: (enter "" to reset)',
                                  curses.A_BOLD)
                    window.addstr(rows - 2, 0,
                                  'eg. "running", ".*", "deleted')
                    window.refresh()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    text_box = Textbox(win)
                    text = text_box.edit(validate=manage_backspace).strip()
                    toggle(args, 'status_filter', text.strip())
                    visualise_results(results, args)
                if c == ord("o"):
                    expand_helps = not expand_helps
                    visualise_results(results, args)

    except KeyboardInterrupt:
        p_request.terminate()
        p_process.terminate()
        raise KeyboardInterrupt


def send_request_task(q_requests, q_response):
    import socket
    from xmen.config import Config
    import time
    config = Config()
    context = ssl.create_default_context()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
            with context.wrap_socket(ss, server_hostname=config.server_host) as s:
                s.connect((config.server_host, config.server_port))
                while True:
                    try:
                        try:
                            request = q_requests.get(False)
                        except queue.Empty:
                            pass

                        if not request:
                            break
                        send(request, s)
                        response = receive(s)
                        q_response.put(response)
                        time.sleep(0.1)
                    except queue.Empty:
                        pass
    except (socket.error, IOError):
        pass
