# ID Manager Agent (`id_manager_agent.py`) - Production Candidate v2

## Introduction

The `IDManagerAgent` is a specialized component within the mindX Augmentic Intelligence. `IDManagerAgent` core function is to generate and manage Ethereum-compatible cryptographic key pairs (consisting of a public address and a private key) for other agents, tools, or any entity within mindX requiring a unique and verifiable digital identity. This capability is foundational for scenarios involving secure inter-agent communication, data signing, or potential interactions with blockchain-based systems.

This version emphasizes secure local storage of private keys within a dedicated `.env` file, namespaced per `IDManagerAgent` instance, and includes more robust key naming and retrieval. The conceptual link to `EmbalmerAgent/Tomb` for file encryption remains an external operational concern beyond this agent's direct implementation, though the agent attempts to set restrictive file permissions.

## Explanation

### Core Functionality

-   **Named Instances & Singleton Access (`get_instance`):**
    -   An asynchronous class method `get_instance(agent_id: str, ...)` serves as a factory. This allows for multiple, uniquely named `IDManagerAgent` instances if needed (e.g., managing distinct sets of keys for different purposes or security domains), though typically one instance per major system component might be sufficient.
    -   Instances are cached by their `agent_id` at the class level. `reset_all_instances_for_testing()` clears this cache.

-   **Secure Environment Setup (`__init__`, `_ensure_env_setup_sync`):**
    *   Each `IDManagerAgent` instance is associated with a specific `agent_id`.
    *   It establishes a dedicated data directory: `PROJECT_ROOT/data/id_manager_work/<sanitized_agent_id>/`. The `agent_id` part is sanitized to be filesystem-friendly.
    *   Within this directory, it manages a specific `.env` file named `.wallet_keys.env` to store the private keys.
    *   On initialization:
        *   The data directory and `.env` file are created if they don't exist.
        *   It attempts to set restrictive file permissions (0600 for `.env` file, 0700 for its directory on POSIX systems) as a basic security measure. This relies on the OS and user privileges.
        *   It loads any existing keys from this specific `.env` file into the current process's environment variables for potential use by `os.getenv()`, using `python-dotenv`.

-   **Wallet Creation (`create_new_wallet`):**
    *   Uses the `eth_account` Python library to generate a new standard Ethereum key pair.
    *   **Private Key Storage:**
        *   The generated private key (hexadecimal string) is stored in the agent's dedicated `.wallet_keys.env` file.
        *   A unique environment variable name is constructed for each key: `MINDX_WALLET_PK_<ENTITY_ID_SLUG>_<PUBLIC_ADDRESS_UPPERCASE>`.
            -   `<ENTITY_ID_SLUG>`: A sanitized version of an optional `entity_id` provided (e.g., the ID of the agent or tool this wallet is for). Defaults to a short UUID if no `entity_id` is given.
            -   `<PUBLIC_ADDRESS_UPPERCASE>`: The full public address in uppercase. Using the full address in the key name improves uniqueness and retrievability.
        *   Uses `dotenv.set_key()` to write the variable to the specific `.env` file. This function also typically reloads the environment variables for the current process.
    *   Returns the `(public_address_str, private_key_hex_str)` tuple.

-   **Private Key Retrieval (`get_private_key`):**
    *   Takes a `public_address` and an optional `entity_id_hint`.
    *   It reloads the agent's `.wallet_keys.env` using `load_dotenv(override=True)` to ensure `os.getenv()` can access the most current set of keys defined in that file.
    *   It first attempts to retrieve the key by reconstructing the precise environment variable name using the `public_address` and `entity_id_hint`.
    *   If the hint was provided and that fails, it tries again assuming a generic `entity_id` ("UNSPECIFIED") was used during creation.
    *   *(Note: The previous broader scan of all `MINDX_WALLET_PK_` env vars is less reliable and less efficient for direct key retrieval if the exact env var name isn't known or reconstructible. The current approach relies on knowing/reconstructing the name.)*
    *   Returns the private key string or `None` if not found.

-   **Identity Deprecation (`deprecate_identity`):**
    *   Takes a `public_address` and `entity_id_hint`.
    *   Attempts to remove the corresponding private key variable from the `.env` file using `dotenv.unset_key()`.
    *   This effectively "retires" the identity from this manager's perspective. It does not securely wipe the key from disk history unless the `.env` file is later shredded.

-   **Listing Managed Identities (`list_managed_identities`):**
    *   Provides a list of potential identities managed by this instance.
    *   It reads the agent's specific `.env` file directly using `dotenv.dotenv_values()` (which doesn't pollute `os.environ`) and parses keys that match the `MINDX_WALLET_PK_` pattern.
    *   Attempts to extract the `entity_id` part and the full `public_address` from the environment variable names.
    *   Returns a list of dictionaries, each containing `{"entity_id_part", "public_address", "env_var_name"}`.

-   **Shutdown (`shutdown`):**
    *   Conceptual placeholder. If a more advanced secure storage mechanism (like an encrypted container via the conceptual `EmbalmerAgent/Tomb`) were used, this method would ensure it's properly closed and locked. For the current `.env` file approach, no specific shutdown actions are strictly necessary beyond normal file system operations.

## Technical Details

-   **Dependencies:**
    -   `python-dotenv`: For reading from and writing to `.env` files.
    -   `eth-account`: The Ethereum Foundation's library for Ethereum account management, including key pair generation.
-   **Path Management:** Uses `pathlib.Path` and `PROJECT_ROOT` (from `mindx.utils.config`) to create a dedicated, namespaced data directory for each `IDManagerAgent` instance (e.g., `PROJECT_ROOT/data/id_manager_work/my_mastermind_id_manager/`).
-   **Security Considerations:**
    -   **Private Key Storage:** Private keys are stored in plain text within the `.wallet_keys.env` file. The agent attempts to set restrictive file permissions (0600 for the file, 0700 for its directory on POSIX systems).
    -   **Encryption Responsibility:** This agent **does not implement file-level encryption** itself. Protecting the `.wallet_keys.env` file relies on strong OS-level file permissions and overall host security. For production, this file should be treated as highly sensitive and might be managed by external secrets management systems (e.g., HashiCorp Vault, cloud provider KMS, Kubernetes Secrets) which would then make keys available to the MindX process environment securely.
    -   **Environment Variables:** Loading private keys into environment variables via `os.getenv` makes them accessible to the current Python process.

## Usage

The `IDManagerAgent` is intended to be used by high-level orchestrators within MindX, such as the `MastermindAgent`, when new agents, tools, or components require a unique, cryptographically verifiable identity.

  **Getting an Instance (Asynchronous Factory Recommended):**<br />
    ```python
    # from mindx.core.id_manager_agent import IDManagerAgent
    # import asyncio

    # async def get_id_manager():
    #     # Get a specific named instance, e.g., for managing identities for all "production" agents
    #     id_manager = await IDManagerAgent.get_instance(agent_id="mindx_production_identities")
    #     return id_manager
    ```

  **Creating a New Wallet (Identity):**<br />
    ```python
    # async def provision_new_agent_identity(id_manager: IDManagerAgent, new_agent_name: str):
    #     try:
    #         public_addr, private_key_hex = id_manager.create_new_wallet(entity_id=new_agent_name)
    #         print(f"Identity for '{new_agent_name}': Address: {public_addr}")
    #         # The public_addr can now be stored as the new agent's identifier.
    #         # The private_key_hex is securely stored by IDManagerAgent; the new agent
    #         # would retrieve it via get_private_key() when needed for signing.
    #         return public_addr
    #     except Exception as e:
    #         print(f"Failed to provision identity for '{new_agent_name}': {e}")
    #         return None
    ```

  **Retrieving a Private Key (by an entity that knows its public address):**<br />
    ```python
    # async def agent_uses_its_key(id_manager: IDManagerAgent, my_public_address: str, my_entity_id: str):
    #     my_private_key = id_manager.get_private_key(my_public_address, entity_id_hint=my_entity_id)
    #     if my_private_key:
    #         print(f"Agent {my_entity_id}: Successfully retrieved my private key.")
    #         # Now use the private key for cryptographic operations
    #     else:
    #         print(f"Agent {my_entity_id}: Could not retrieve my private key!")
    ```

  **Listing Identities:**<br />
    ```python
    # identities = id_manager.list_managed_identities()
    # for identity_info in identities:
    #     print(f"- Entity Hint: {identity_info['entity_id_part']}, PubAddr: {identity_info.get('public_address', identity_info.get('public_address_suffix'))}")
    ```
<br /><br />
The `IDManagerAgent` provides a foundational service for identity management in mindX, enabling more advanced scenarios where components need to prove their identity or securely sign data. Its integration with the `MastermindAgent` allows for strategic provisioning of identities as the MindX system evolves or spawns new entities.
