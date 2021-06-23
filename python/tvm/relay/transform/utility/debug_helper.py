import sys

# With more argumentsc
def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# def printe(msg):
#     print(msg, file=sys.stderr)
