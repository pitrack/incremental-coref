import sys
import _jsonnet
import json

if len(sys.argv) > 1:
    check_file = sys.argv[1]
else:
    check_file = "../experiments.jsonnet"

f = json.loads(_jsonnet.evaluate_file(check_file))
# print (f.keys())

print (f"OK! {len(f)} configs loaded! ({sys.getsizeof(f)} bytes)")
