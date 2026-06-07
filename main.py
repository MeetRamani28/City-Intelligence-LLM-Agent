import os
from dotenv import load_dotenv
from rich import print

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

from tools import get_weather, get_news

load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every autonomous tool execution."""
    tool_name = request.tool_call["name"]
    print(f"\n[bold yellow]⚠️  [AGENT SYSTEM]: Wants to access terminal tool '{tool_name}'[/bold yellow]")
    confirm = input("Confirm execution? (yes/no): ")

    if confirm.lower() != "yes":
        print("[bold red]❌ Tool execution hijacked and blocked by supervisor.[/bold red]\n")
        return ToolMessage(
            content="Tool call denied by supervisor user.",
            tool_call_id=request.tool_call["id"]
        )

    print("[bold green]✅ Approved. Running node pipeline...[/bold green]\n")
    return handler(request)  

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful and polite city assistant. Use tools when specific metrics are asked.",
    middleware=[human_approval]
)

if __name__ == "__main__":
    print("\n[bold cyan]🏙️  City Intelligence Agent Active | Type 'exit' to safe quit[/bold cyan]\n")
    
    while True:
        user_input = input("You : ")
        if user_input.lower() == "exit":
            print("[bold gray]Goodbye![/bold gray]")
            break 
            
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            print("\n[bold green]Bot :[/bold green]", result['messages'][-1].content, "\n")
        except Exception as e:
            print(f"[bold red]Execution Interrupted: {e}[/bold red]\n")