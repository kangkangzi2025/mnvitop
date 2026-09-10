"""MCP tool functions, exercised without a transport by patching the probe."""

import mnvitop.mcp as mcp_module
from mnvitop.config import HostSpec, Settings
from mnvitop.model import Snapshot
from mnvitop.probe import parse_probe_output
from tests.test_probe import SAMPLE


async def _fake_probe(spec, timeout, control_dir, script=None):
    if spec.name == "down":
        return Snapshot(host=spec.name, error="ssh: connect timed out")
    return parse_probe_output(spec.name, SAMPLE)


def _settings():
    return Settings(hosts=[HostSpec("a", "a"), HostSpec("down", "down")])


async def test_tools(monkeypatch):
    monkeypatch.setattr(mcp_module, "run_probe", _fake_probe)
    monkeypatch.setattr(mcp_module, "load_settings", _settings)
    hosts = await mcp_module.list_hosts()
    assert [host["name"] for host in hosts["hosts"]] == ["a", "down"]
    status = await mcp_module.fleet_status()
    assert status["summary"].startswith("1/2 hosts up")
    assert status["hosts"][1]["ok"] is False
    only = await mcp_module.fleet_status(hosts=["a", "nope"], processes=False)
    assert [host["host"] for host in only["hosts"]] == ["a", "nope"]
    assert "not a configured host" in only["hosts"][1]["error"]
    placement = await mcp_module.find_gpus(count=1)
    assert placement["found"] and placement["placements"][0]["gpus"] == [2]


async def test_server_builds_with_three_tools():
    server = mcp_module.build_server()
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {"list_hosts", "fleet_status", "find_gpus"}
    assert tools["find_gpus"].description.startswith("Find where a job")
    schema = getattr(tools["find_gpus"], "input_schema", None) or getattr(tools["find_gpus"], "inputSchema")
    assert set(schema["properties"]) == {"count", "min_free_gb", "hosts"}


async def test_no_hosts_configured(monkeypatch):
    monkeypatch.setattr(mcp_module, "load_settings", lambda: Settings(hosts=[]))
    assert (await mcp_module.list_hosts())["hosts"] == []
    assert "no hosts configured" in (await mcp_module.fleet_status())["summary"]
    assert (await mcp_module.find_gpus())["found"] is False


async def test_stdio_server_with_real_client(monkeypatch):
    import json
    import os
    import sys

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = dict(os.environ, MNVITOP_HOSTS="here=local", MNVITOP_CONFIG="/nonexistent/hosts.toml")
    server = StdioServerParameters(command=sys.executable, args=["-m", "mnvitop.mcp"], env=env)
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert names == {"list_hosts", "fleet_status", "find_gpus"}
            hosts = json.loads((await session.call_tool("list_hosts", {})).content[0].text)
            assert hosts["hosts"] == [{"name": "here", "ssh": "local"}]
            status = json.loads((await session.call_tool("fleet_status", {"processes": False})).content[0].text)
            assert status["hosts"][0]["host"] == "here" and "summary" in status
            place = json.loads((await session.call_tool("find_gpus", {"count": 1})).content[0].text)
            assert set(place) >= {"found", "summary", "placements", "alternatives"}
