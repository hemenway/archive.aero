#!/bin/zsh
# Rebuild the archive-slicer image on the OMV NUC and (re)start its stack.
#   slicerd/deploy.sh          sync build context, build, push compose.yml into
#                              the OMV compose plugin, `up -d`
# A running job dies with the old container: check `slicerctl status` first.
set -euo pipefail
HOST=${SLICERD_SSH:-omv}
HERE=${0:A:h}
STACK=archive-slicer

rsync -a --delete --exclude mac/ --exclude __pycache__ "$HERE/" "$HOST:/srv/archive-slicer/build/"
# host helper: stop + start the container when the controller asks for a fresh rawtiffs overlay
ssh "$HOST" "install -m 644 /srv/archive-slicer/build/host/archive-slicer-remount.* /etc/systemd/system/ \
  && systemctl daemon-reload && systemctl enable --now archive-slicer-remount.path >/dev/null 2>&1 && echo remount helper enabled"
ssh "$HOST" "cd /srv/archive-slicer/build && docker build -q -t archive-slicer:latest . >/dev/null && echo built"

# The compose plugin's DB entry is the source of truth for the stack file
# (it regenerates /compose/$STACK/$STACK.yml from it).
python3 - "$HERE/compose.yml" <<'EOF' | ssh "$HOST" 'python3 /dev/stdin'
import json, sys
body = open(sys.argv[1]).read()
print(f"""
import json, subprocess
body = {body!r}
files = json.loads(subprocess.check_output(["omv-confdbadm", "read", "conf.service.compose.file"]))
cur = next((f for f in files if f["name"] == "{'archive-slicer'}"), None)
params = {{"name": "archive-slicer", "description": "archive.aero slicer + geotiff2pmtiles job server (MCP/REST :8765)",
           "body": body, "showenv": False, "env": "", "showoverride": False, "override": ""}}
if cur:
    params["uuid"] = cur["uuid"]  # no uuid = create
if cur and cur["body"] == body:
    print("compose file unchanged")
else:
    subprocess.run(["omv-rpc", "-u", "admin", "Compose", "setFile", json.dumps(params)], check=True,
                   stdout=subprocess.DEVNULL)
    print("compose file " + ("updated" if cur else "created"))
""")
EOF

ssh "$HOST" "cd /compose/$STACK && docker compose --project-name $STACK --project-directory /compose/$STACK \
  -f $STACK.yml \$( [ -f compose.override.yml ] && echo -f compose.override.yml ) \
  --env-file /compose/global.env \$( [ -f $STACK.env ] && echo --env-file $STACK.env ) up -d --force-recreate 2>&1 | tail -3"
sleep 4
curl -fsS "http://192.168.1.203:8765/health" && echo " ← archive-slicer up"
