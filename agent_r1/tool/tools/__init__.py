def _default_tool(name):
    print("***********name*************:",name)
    if name == "prompt":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "multi_llm_tool":
        from agent_r1.tool.tools.APIYI_tool import MultiLLMTool
        return MultiLLMTool()
    elif name == "gpt4omini_tool":
        from agent_r1.tool.tools.API_tool import GPT4oMiniTool
        return GPT4oMiniTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")