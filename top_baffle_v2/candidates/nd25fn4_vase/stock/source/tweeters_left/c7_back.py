import json, sys
C = json.load(open(sys.argv[1]))
neg = lambda p: [-p[0], *p[1:]]
for k in ("UM", "T", "t"): C[k] = neg(C[k])
C["lanes"] = {k: [neg(p) for p in v] for k, v in C["lanes"].items()}
C["ports"]["x"] *= -1; C["ports"]["lm_duct"]["x"] *= -1
sl = C["ports"]["slot"]; sl["x_out"], sl["x_in"] = -sl["x_out"], -sl["x_in"]
C["plan"] = [neg(p) for p in C["plan"]]; C["hand"] = "tweeters_left (design frame)"
json.dump(C, open(sys.argv[2], "w")); print("c7 back to the design frame:", C["T"], C["ports"]["lm_duct"])
