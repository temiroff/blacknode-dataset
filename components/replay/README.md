# Replay

Component of `blacknode-dataset`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="replay", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.replay]
    nodes = ["components/replay/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
