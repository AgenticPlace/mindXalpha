# mindX Core: Belief System

The Belief System is a fundamental component of agents within the mindX (Augmentic Intelligence) framework. It provides a structured way for agents to store, manage, and reason about information they acquire from perceptions, communications, inferences, or other sources.

## Core Concepts

###  `BeliefSource` (Enum)

Defines the origin or nature of a piece of information (a belief). This helps the agent (and the system) understand the trustworthiness or context of the belief.

**Defined Sources:**

*   `PERCEPTION`: Information directly observed from the environment or internal sensors.
*   `COMMUNICATION`: Information received from another agent, user, or external system.
*   `INFERENCE`: Information deduced or derived logically from other existing beliefs.
*   `SELF_ANALYSIS`: Information derived from the agent reflecting on its own internal state or past actions.
*   `EXTERNAL_INPUT`: Information explicitly provided by a human user or a non-agent external system.
*   `DEFAULT`: A pre-configured, fallback, or assumed value.
*   `LEARNED`: Information acquired through a machine learning process or direct experience-based adaptation.
*   `DERIVED`: A more general category for information processed or transformed from other data, often by an internal cognitive action or tool.

###  `Belief` (Class)

Represents an individual piece of information held by an agent. Each belief has:

*   `value`: The actual data or content of the belief (can be any Python type).
*   `confidence`: A float between 0.0 and 1.0 indicating the agent's certainty in this belief.
*   `source`: An instance of `BeliefSource` indicating where this belief came from.
*   `timestamp`: The time (Unix timestamp) when the belief was initially created.
*   `last_updated`: The time when the belief's value, confidence, or source was last changed.

**Methods:**
*   `update(new_value, new_confidence, new_source)`: Modifies the belief's attributes.
*   `to_dict()`: Serializes the belief into a dictionary (useful for persistence).
*   `from_dict(data)`: Deserializes a belief from a dictionary.

###  `BeliefSystem` (Class)

The central manager for an agent's beliefs. It's typically implemented as a singleton to ensure a consistent belief store for a given agent instance or context.

**Key Features:**

*   **Storage:** Stores beliefs in a dictionary where keys are strings (often namespaced, e.g., `"environment.temperature"`, `"bdi.my_domain.beliefs.current_task"`) and values are `Belief` objects.
*   **Persistence (Optional):** Can load beliefs from and save beliefs to a JSON file if a `persistence_file_path` is provided during initialization. This allows beliefs to persist across agent restarts.
*   **Thread/Async Safety:** Uses locks to manage concurrent access to the belief store, making it suitable for multi-threaded or asynchronous agents (the example uses `threading.Lock`; for purely async agents, `asyncio.Lock` would be more appropriate for async methods).
*   **Singleton Pattern:** Ensures only one instance of the belief system exists (per context, if multiple belief systems are needed for different agent domains).

**Core Methods (asynchronous):**

*   `async def update_belief(key, value, confidence, source, metadata, ttl_seconds)`:
    *   Adds a new belief or updates an existing one.
    *   `key`: String identifier for the belief.
    *   `value`: The content of the belief.
    *   `confidence`: Certainty (0.0-1.0).
    *   `source`: A `BeliefSource` enum member.
    *   `metadata` (Optional): Additional dictionary for context (not fully utilized in the basic `Belief` class example but available).
    *   `ttl_seconds` (Optional): Time-to-live for the belief (not fully implemented in the basic `Belief` class example but available for extension).
*   `async def add_belief(...)`: Alias for `update_belief`.
*   `async def get_belief(key)`: Retrieves the `Belief` object for a given key, or `None`. Returns a deep copy to prevent external modification of the internal `Belief` object.
*   `async def get_belief_value(key, default=None)`: Retrieves just the `value` of a belief, or a default if not found.
*   `async def remove_belief(key)`: Deletes a belief.
*   `async def get_all_beliefs()`: Returns a deep copy of the entire dictionary of beliefs.
*   `async def query_beliefs(partial_key, min_confidence, source)`: Allows searching for beliefs based on parts of their key, minimum confidence, and/or source.

**Internal Methods:**
*   `_load_beliefs()`: Loads beliefs from the JSON persistence file.
*   `_save_beliefs()`: Saves current beliefs to the JSON persistence file.
*   `_save_beliefs_if_path()`: Helper to save only if a persistence path is configured.

**Singleton Management:**
*   `__new__`: Implements the singleton creation logic.
*   `reset_singleton_for_testing()`: A class method to clear the singleton instance, useful for isolated tests.

## Usage

### Initialization

The `BeliefSystem` is typically initialized once per agent or logical context. If persistence is desired, a file path is provided.

```python
from core.belief_system import BeliefSystem, BeliefSource, Belief
from pathlib import Path

# Example: Initialize with persistence
# PROJECT_ROOT would typically be defined in your utils.config
belief_store_path = PROJECT_ROOT / "data" / "my_agent_beliefs.json"
bs = BeliefSystem(persistence_file_path=belief_store_path)

# Example: Initialize without persistence (in-memory only for the session)
# bs_in_memory = BeliefSystem()

# Updating and Adding Beliefs
Beliefs are added or updated using the update_belief (or add_belief) method.
```python
import asyncio

async def manage_beliefs():
    # Assuming 'bs' is an initialized BeliefSystem instance
    await bs.update_belief(
        key="environment.is_light_on",
        value=True,
        confidence=0.95,
        source=BeliefSource.PERCEPTION
    )

    await bs.update_belief(
        key="user.preference.theme",
        value="dark",
        confidence=1.0,
        source=BeliefSource.EXTERNAL_INPUT
    )

    # Updating an existing belief
    await bs.update_belief(
        key="environment.is_light_on",
        value=False,
        confidence=0.98, # Confidence might change with new perception
        source=BeliefSource.PERCEPTION
    )

# asyncio.run(manage_beliefs())
```
# Retrieving Beliefs
You can get the full Belief object or just its value.
```python
async def check_beliefs():
    light_status_belief = await bs.get_belief("environment.is_light_on")
    if light_status_belief:
        print(f"Light is on: {light_status_belief.value} (Confidence: {light_status_belief.confidence})")
        print(f"Source: {light_status_belief.source.value}, Updated: {time.ctime(light_status_belief.last_updated)}")

    theme_preference = await bs.get_belief_value("user.preference.theme", default="light")
    print(f"User theme preference: {theme_preference}")

    unknown_value = await bs.get_belief_value("non.existent.key")
    print(f"Unknown key value: {unknown_value}") # Will be None (default for get_belief_value)
```


# Removing Beliefs
```python
async def clear_belief():
    await bs.remove_belief("environment.is_light_on")
    print("Light status belief removed (if it existed).")
```
# Querying Beliefs
The query_beliefs method allows for more flexible retrieval
```python
async def query_example():
    # Get all beliefs related to the environment
    env_beliefs = await bs.query_beliefs(partial_key="environment.")
    print("Environment Beliefs:")
    for belief_obj in env_beliefs:
        # Note: query_beliefs in the example returns Belief objects,
        # which might have a .key attribute if the BeliefSystem stores them that way internally
        # or if the query_beliefs method reconstructs it.
        # The provided BeliefSystem example stores them by key, so this would work.
        print(f"  {getattr(belief_obj, 'key', 'Unknown Key')}: {belief_obj.value}")

    # Get high-confidence perceptions
    high_conf_perceptions = await bs.query_beliefs(min_confidence=0.9, source=BeliefSource.PERCEPTION)
    print("High Confidence Perceptions:")
    for belief_obj in high_conf_perceptions:
        print(f"  Value: {belief_obj.value}, Confidence: {belief_obj.confidence}")
```
# Persistence
If a persistence_file_path is provided to the BeliefSystem constructor:<br /><br />
Beliefs are loaded from this file upon initialization (if the file exists)<br /><br />
Beliefs are saved to this file whenever a belief is updated or removed<br /><br />
This allows the agent's knowledge to persist across sessions. The file is stored in JSON format<br /><br />
# Integration with BDI Agents
The BDIAgent (in core.bdi_agent.py) uses an instance of BeliefSystem<br /><br />
It namespaces its beliefs (e.g., bdi.<domain_name>.beliefs.some_key)<br /><br />
Its internal logic (perceiving, planning, executing actions) frequently interacts with the belief system to read current state and update its understanding of the world or task<br /><br />
For example, the outcome of an LLM call during planning or a tool execution result is often stored as a new belief<br /><br />
# Considerations and Future Extensions
Asynchronous Operations: All belief modification and retrieval methods are async to fit into an asynchronous agent architecture. If used in a purely synchronous context without an event loop, asyncio.run() would be needed for each call, or the methods could be adapted<br /><br />
Locking: The current example uses threading.Lock. For a fully asyncio-based system, asyncio.Lock should be used with async with self._async_lock: in the methods<br /><br />
Complex Queries: The current query_beliefs is basic. More advanced query languages or mechanisms (e.g., based on belief content, time ranges) could be added<br /><br />
Belief Revision/Truth Maintenance: More sophisticated belief systems include logic for belief revision (handling contradictory information) and truth maintenance (retracting beliefs that depend on other retracted beliefs). This is a complex area not covered by the current simple implementation<br /><br />
Time-To-Live (TTL): The update_belief signature includes ttl_seconds, hinting at future support for beliefs that automatically expire. This would require a background process or checks during retrieval<br /><br />
Scalability: For a very large number of beliefs or high-frequency updates, an in-memory dictionary might become a bottleneck. External databases (NoSQL, graph databases) could be considered for larger-scale systems<br /><br />
Integration with Config for Persistence Path: The persistence path could be made configurable through the main Config system instead of being passed directly at instantiation, or PROJECT_ROOT could be injected<br /><br />
This Belief System provides a solid foundation for knowledge representation and management within the mindX agents<br /><br />
