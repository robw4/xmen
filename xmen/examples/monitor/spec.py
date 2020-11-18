from xmen.monitor import Spec

# full modulo string
config = Spec('^xx|yy@100s')
print(config)

# modulo strings can have an empty regex
config = Spec('@10s')
print(config)

# modulo strings *must* have both a steps and trigger
try:
    config = Spec('a random string')
except AssertionError as m:
    print(m)
try:
    config = Spec('xx@10')
except AssertionError as m:
    print(m)
try:
    config = Spec('xx@s')
except AssertionError as m:
    print(m)
