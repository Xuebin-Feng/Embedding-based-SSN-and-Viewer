import Command_Engine
def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: redo\nDescription: Reapplies a state that was previously undone using the `undo` command."
        Command_Engine.print_help(viewer, msg)
        return
    viewer._do_redo()
