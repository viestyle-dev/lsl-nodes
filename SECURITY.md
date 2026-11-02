# Security Policy

## Reporting a Vulnerability

Please use GitHub's **Private Vulnerability Reporting**
(Security tab → "Report a vulnerability") on this repository.
Do not open public issues for security reports.

## Scope

lsl-nodes publishes and consumes Lab Streaming Layer streams. `LslOutlet`
announces a stream on the local network through LSL discovery, so any LSL
client on the network can subscribe to it. `LslInlet` connects to the
first stream that matches the configured name/type without further
authentication, so a peer on the same network can present a stream under
the same name. `LslStreamResolver` exposes the host names of discovered
streams to the graph. Use liblsl's `lsl_api.cfg` to restrict discovery to
known hosts or multicast scopes. The plugin does not execute code from
streams or graph definitions.

The host runtime it plugs into (the nonoha engine) may include script
nodes with their own threat model — see the security policy of the host
project for loading untrusted graphs.

## Supported versions

Security fixes target the latest minor release.
