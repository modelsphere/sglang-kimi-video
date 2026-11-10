"""Load plugins in each freshly spawned HTTP/tokenizer worker.

The pinned SGLang multi-worker HTTP import path does not call load_plugins.
Select this module through SGLANG_HTTP_SERVER_APP_PATH for that deployment.
"""

from sglang.srt.entrypoints.http_server import app as app
from sglang.srt.plugins import load_plugins

from .compat import check_targets

load_plugins()
check_targets(applied=True)
