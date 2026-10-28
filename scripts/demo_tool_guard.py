"""Enforce the user-selected tool menu before the existing demo executor runs."""
def execute_selected_tool(call, tools, executor):
    if not isinstance(call,dict):return {'error':'Invalid tool call'}
    enabled={t['function']['name'] for t in tools or []}
    if call.get('name') not in enabled:return {'error':'Tool is not enabled in the selected menu'}
    arguments=call.get('arguments',{})
    if not isinstance(arguments,dict):return {'error':'Tool arguments must be a JSON object'}
    return executor(call['name'],arguments)
