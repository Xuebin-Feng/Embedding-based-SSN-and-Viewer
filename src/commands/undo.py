import Command_Engine
def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: undo\nDescription: Reverts the visual and spatial state of the network to the previous action.\nMost commands automatically save state before execution, allowing them to be undone."
        Command_Engine.print_help(viewer, msg)
        return
    viewer._do_undo()
