"""Isolation probe for wls_mbpp_sandbox.sh. Executes no generated code.

Run with the identical sandbox command used for scoring; prints one JSON object.
wls_preflight_dryrun.py turns it into pass/fail checks.
"""
import hashlib
import json
import multiprocessing as mp
import os
import socket
import sys



def main():
    res = {}
    st = {line.split(":")[0]: line.split(":")[1].strip() for line in open("/proc/self/status") if ":" in line}
    res["uid_gid"] = [os.getuid(), os.geteuid(), os.getgid()]
    res["caps"] = {k: st[k] for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs")}
    res["net_ifaces"] = [n for _, n in socket.if_nameindex()]


    def tryconn(h, p):
        try:
            socket.create_connection((h, p), timeout=3).close()
            return "CONNECTED"
        except Exception as e:
            return f"blocked: {type(e).__name__}"


    res["tcp"] = {"1.1.1.1:443": tryconn("1.1.1.1", 443), "8.8.8.8:53": tryconn("8.8.8.8", 53)}
    try:
        socket.getaddrinfo("github.com", 443)
        res["dns"] = "RESOLVED"
    except Exception as e:
        res["dns"] = f"blocked: {type(e).__name__}"

    host_paths = sys.argv[1].split(":") if len(sys.argv) > 1 else []
    res["host_paths_visible"] = {p: os.path.exists(p) for p in
                                 host_paths + ["/home", "/mnt", "/mnt/c", "/root", "/etc/passwd",
                                               "/etc/shadow", "/run", "/var", "/media", "/srv", "/opt"]}


    def trywrite(p):
        try:
            with open(p, "w") as f:
                f.write("x")
            os.remove(p)
            return "WRITABLE"
        except Exception as e:
            return f"denied: {type(e).__name__}"


    res["write"] = {p: trywrite(p) for p in ["/x", "/usr/x", "/sbx/venv/x", "/sbx/code/evaluation/x",
                                             "/sbx/data/MBPP/x", "/sbx/in/x", "/sbx/run/x", "/etc/x",
                                             "/tmp/x", "/dev/x"]}
    try:
        open("/sbx/data/MBPP/test.json", "a").close()
        res["append_test_json"] = "WRITABLE"
    except Exception as e:
        res["append_test_json"] = f"denied: {type(e).__name__}"
    try:
        open("/tmp/x2", "w").close()
        os.replace("/tmp/x2", "/sbx/data/MBPP/test.json")
        res["replace_test_json"] = "REPLACED"
    except Exception as e:
        res["replace_test_json"] = f"denied: {type(e).__name__}"
    for name, flag in [("nested_userns", os.CLONE_NEWUSER), ("new_mount_ns", os.CLONE_NEWNS)]:
        try:
            os.unshare(flag)
            res[name] = "ALLOWED"
        except Exception as e:
            res[name] = f"denied: {type(e).__name__}"

    # Import the evaluator package first, exactly as the harness does (it sets the start method).
    sys.path.insert(0, "/sbx/code")
    import evaluation.MBPP.mbpp  # noqa: E402,F401
    import numpy  # noqa: E402

    q = mp.Manager().list()
    p = mp.Process(target=q.append, args=(1,))
    p.start()
    p.join()
    res["multiprocessing_ok"] = list(q) == [1]
    res["env_keys"] = sorted(os.environ)
    res["test_json_sha256"] = hashlib.sha256(open("/sbx/data/MBPP/test.json", "rb").read()).hexdigest()
    res["python"] = sys.version.split()[0]
    res["numpy"] = numpy.__version__
    res["start_method"] = mp.get_start_method()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
