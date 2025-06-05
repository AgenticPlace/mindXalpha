# mindx/core/id_manager_agent.py
"""
IDManagerAgent for MindX.

Manages Ethereum-compatible cryptographic identities (wallet addresses and private keys)
for agents, tools, or other components within the MindX system. This provides a
foundation for secure, verifiable identification.

Private keys are stored in a dedicated .env file within a secure, agent-specific
data directory. The actual encryption of this .env file at rest (e.g., via Tomb
or OS-level full-disk encryption) is an external operational security measure.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import re
import time
import uuid
import asyncio # For async lock in factory

# Using python-dotenv for .env file management
# Requires: pip install python-dotenv eth_account
try:
    from dotenv import load_dotenv, set_key, unset_key, find_dotenv
    from eth_account import Account # type: ignore
    # Enable HD Wallet features if you plan to use mnemonic-based generation or derivation.
    # For simple random account creation, it's not strictly necessary but doesn't hurt.
    if hasattr(Account, 'enable_unaudited_hdwallet_features'): # pragma: no cover
        Account.enable_unaudited_hdwallet_features()
except ImportError as e: # pragma: no cover
    # This allows the module to be imported even if dependencies are missing,
    # but functionality will be severely limited.
    logging.getLogger(__name__).critical(
        f"IDManagerAgent dependencies missing: {e}. Please run 'pip install python-dotenv eth-account'. "
        "IDManagerAgent will not function correctly."
    )
    # Define Account as a dummy to prevent further import errors in type hints / stubs
    class Account: # type: ignore
        address: str
        key: Any # Actually a LocalAccount.privateKey which is HexBytes
        @staticmethod
        def create() -> 'Account': raise NotImplementedError("eth_account library not installed")
    def load_dotenv(*args, **kwargs): pass # type: ignore 
    def set_key(*args, **kwargs) -> bool: return False # type: ignore
    def unset_key(*args, **kwargs) -> bool: return False # type: ignore
    def find_dotenv(*args, **kwargs) -> str: return "" # type: ignore

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class IDManagerAgent:
    """
    Manages Ethereum-compatible wallet creation and secure storage of private keys
    within a dedicated, permission-restricted .env file.
    """
    _instances: Dict[str, 'IDManagerAgent'] = {} # Cache for named instances
    _class_lock = asyncio.Lock() # For async-safe singleton factory

    @classmethod
    async def get_instance(cls, 
                           agent_id: str = "default_identity_manager", 
                           config_override: Optional[Config] = None,
                           test_mode: bool = False) -> 'IDManagerAgent': # pragma: no cover
        """Factory method to get or create a named instance of IDManagerAgent asynchronously."""
        async with cls._class_lock:
            if agent_id not in cls._instances or test_mode:
                if test_mode and agent_id in cls._instances: # pragma: no cover
                    logger.debug(f"IDManagerAgent Factory: Resetting instance for '{agent_id}' due to test_mode.")
                    old_instance = cls._instances.pop(agent_id, None)
                    if old_instance and hasattr(old_instance, 'shutdown') and asyncio.iscoroutinefunction(old_instance.shutdown):
                        await old_instance.shutdown()
                
                logger.info(f"IDManagerAgent Factory: Creating new instance for ID '{agent_id}'.")
                instance = cls(
                    agent_id=agent_id, 
                    config_override=config_override,
                    _is_factory_called=True # Internal flag
                )
                cls._instances[agent_id] = instance
            return cls._instances[agent_id]

    def __init__(self, 
                 agent_id: str = "default_identity_manager", 
                 config_override: Optional[Config] = None,
                 _is_factory_called: bool = False): # pragma: no cover
        
        if not _is_factory_called and agent_id not in IDManagerAgent._instances :
             logger.warning(f"IDManagerAgent direct instantiation for '{agent_id}'. Prefer using `await IDManagerAgent.get_instance(...)`.")

        # Prevent re-initialization if it's a managed singleton being fetched again after first creation by factory
        if hasattr(self, '_initialized') and self._initialized and \
           agent_id in IDManagerAgent._instances and IDManagerAgent._instances[agent_id] is self:
            return

        self.agent_id = agent_id
        self.config = config_override or Config() # Use global Config if not overridden
        self.log_prefix = f"IDManagerAgent ({self.agent_id}):"

        # Define dedicated data directory for this ID manager instance
        # Example config key: "id_manager.default_id_manager.data_dir_relative_to_project"
        data_dir_rel_str = self.config.get(f"id_manager.{self.agent_id}.data_dir_relative_to_project", 
                                       f"data/id_manager_work/{self.agent_id.replace(':', '_')}") # Sanitize for dir name
        self.data_dir: Path = PROJECT_ROOT / data_dir_rel_str
        self.env_file_path: Path = self.data_dir / ".wallet_keys.env" # Specific name for this agent's .env

        self._ensure_env_setup_sync() # __init__ is sync, so call sync helper
        logger.info(f"{self.log_prefix} Initialized. Secure .env path: {self.env_file_path}")
        self._initialized = True

    def _ensure_env_setup_sync(self): # pragma: no cover
        """Synchronously ensures the data directory and .env file exist, and loads it."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt': # POSIX systems
                try: os.chmod(self.data_dir, stat.S_IRWXU) # rwx for owner only
                except OSError as e_perm_dir: logger.warning(f"{self.log_prefix} Could not set restrictive permissions on data directory {self.data_dir}: {e_perm_dir}")
            
            if not self.env_file_path.exists():
                self.env_file_path.touch() # Create empty file
                if os.name != 'nt':
                    try: os.chmod(self.env_file_path, stat.S_IRUSR | stat.S_IWUSR) # rw for owner only (0600)
                    except OSError as e_perm_file: logger.warning(f"{self.log_prefix} Could not set restrictive permissions on new .env file {self.env_file_path}: {e_perm_file}")
                logger.info(f"{self.log_prefix} Created new secure .env file at {self.env_file_path}")
            elif os.name != 'nt': # Ensure existing file has restricted perms if possible
                 try: os.chmod(self.env_file_path, stat.S_IRUSR | stat.S_IWUSR)
                 except OSError: pass # Ignore if cannot change, might not be owner

            # Load existing keys from this specific .env file into os.environ for this process
            # `override=False` means it won't overwrite existing os.environ vars if they clash,
            # but for a dedicated secrets file, `override=True` might be safer to ensure file is source of truth.
            # However, `set_key` below will write and can cause `os.environ` to update if library reloads.
            # For now, let's load and let set_key handle updates.
            if load_dotenv(dotenv_path=self.env_file_path, override=False): 
                logger.debug(f"{self.log_prefix} Loaded keys from {self.env_file_path} into current process environment.")
            else:
                logger.debug(f"{self.log_prefix} No keys initially loaded from {self.env_file_path} (file might be empty or just created).")

        except Exception as e: # pragma: no cover
            logger.error(f"{self.log_prefix} CRITICAL error during secure environment setup for {self.env_file_path}: {e}", exc_info=True)
            raise # Re-raise as this is critical for the agent's function

    def _generate_env_var_name(self, public_address: str, entity_id: Optional[str] = None) -> str:
        """Generates a consistent environment variable name for storing a private key."""
        safe_entity_id_part = re.sub(r'\W+', '_', entity_id).upper() if entity_id else "UNSPECIFIED"
        # Using full public address for uniqueness and direct lookup if address is known.
        # Env var names are typically uppercase.
        return f"MINDX_WALLET_PK_{safe_entity_id_part}_{public_address.upper()}"

    def create_new_wallet(self, entity_id: Optional[str] = None, overwrite_if_exists: bool = False) -> Tuple[str, str]:
        """
        Creates a new Ethereum-compatible wallet (address and private key).
        The private key is stored in this agent's dedicated .env file.

        Args:
            entity_id: An optional identifier for the entity this wallet is for (e.g., another agent's ID).
                       Used to construct a more descriptive environment variable name.
            overwrite_if_exists: If True and a key for this public_address (based on naming convention)
                                 already exists, it will be overwritten. Default is False.

        Returns:
            A tuple (public_address: str, private_key_hex: str).
        
        Raises:
            RuntimeError: If eth_account fails or key storage fails.
        """
        logger.info(f"{self.log_prefix} Creating new wallet. Entity association: {entity_id or 'general_purpose'}.")
        try:
            # Account.create() is synchronous.
            account = Account.create() 
        except Exception as e: # pragma: no cover
            logger.error(f"{self.log_prefix} Failed to create eth_account object: {e}. Is 'eth_account' library correctly installed and functional?", exc_info=True)
            raise RuntimeError(f"eth_account.create() failed: {e}") from e

        private_key_hex: str = account.key.hex()
        public_address: str = account.address

        env_var_name = self._generate_env_var_name(public_address, entity_id)

        # Check if key already exists and if overwrite is allowed
        # Need to reload .env to check current state before writing with set_key
        current_env_pk = None
        if find_dotenv(self.env_file_path, usecwd=False, raise_error_if_not_found=False): # Check if file has content
            load_dotenv(dotenv_path=self.env_file_path, override=True) # Ensure os.environ is fresh from this file
            current_env_pk = os.getenv(env_var_name)

        if current_env_pk and not overwrite_if_exists: # pragma: no cover
            logger.warning(f"{self.log_prefix} Wallet key {env_var_name} for {public_address} already exists and overwrite is False. Returning existing (or potentially different if hash collision).")
            # This is tricky: if Account.create() produced the *same* address, it means we're lucky.
            # If it produced a *different* address but our naming convention led to the same env_var_name,
            # that's a problem with the naming or entity_id.
            # For now, assume if current_env_pk exists, it's for *this* public_address.
            return public_address, current_env_pk # Return the one from env

        # Store in the specific .env file using python-dotenv's set_key
        # set_key modifies the file and then reloads the environment.
        if set_key(self.env_file_path, env_var_name, private_key_hex, quote_mode='never'):
            logger.info(f"{self.log_prefix} Stored private key for {public_address} (Entity: {entity_id or 'N/A'}) as {env_var_name} in {self.env_file_path}.")
        else: # pragma: no cover 
            # This fallback should ideally not be needed if set_key works.
            logger.error(f"{self.log_prefix} python-dotenv set_key failed for {env_var_name} in {self.env_file_path}. THIS IS UNEXPECTED.")
            raise RuntimeError(f"Failed to store private key for {public_address} using set_key.")

        return public_address, private_key_hex

    def get_private_key(self, public_address: str, entity_id_hint: Optional[str] = None) -> Optional[str]: # pragma: no cover
        """
        Retrieves the private key for a given public address by reconstructing its expected env var name.
        """
        logger.debug(f"{self.log_prefix} Attempting to retrieve private key for address: {public_address} (Hint: {entity_id_hint})")
        # Ensure current .env state is loaded into os.environ for this process
        load_dotenv(dotenv_path=self.env_file_path, override=True) 

        # Try precise name first if entity_id_hint is good
        env_var_name_attempt = self._generate_env_var_name(public_address, entity_id_hint)
        private_key = os.getenv(env_var_name_attempt)
        if private_key:
            logger.info(f"{self.log_prefix} Retrieved private key for {public_address} using env var '{env_var_name_attempt}'.")
            return private_key

        # Fallback: If entity_id_hint was None or incorrect, try with "UNSPECIFIED" entity_id
        if entity_id_hint is not None: # Only try fallback if a hint was given and failed
            env_var_name_fallback = self._generate_env_var_name(public_address, None) # Uses "UNSPECIFIED"
            private_key = os.getenv(env_var_name_fallback)
            if private_key:
                logger.info(f"{self.log_prefix} Retrieved private key for {public_address} using fallback env var '{env_var_name_fallback}'.")
                return private_key
        
        # Broader scan (less efficient, more of a last resort or diagnostic)
        # Not ideal for direct retrieval due to potential for many keys.
        # This would be better for a list_wallets_with_details method.
        logger.warning(f"{self.log_prefix} Private key not found for public address: {public_address} with hint '{entity_id_hint}' using direct name construction.")
        return None

    def deprecate_identity(self, public_address: str, entity_id_hint: Optional[str] = None, reason: str = "deprecated") -> bool: # pragma: no cover
        """
        "Deprecates" an identity by removing its private key from the .env file.
        The actual key file for an encrypted store would be archived or securely deleted.
        For .env, it means removing the variable.
        """
        logger.warning(f"{self.log_prefix} Attempting to deprecate identity for address: {public_address} (Hint: {entity_id_hint}). Reason: {reason}")
        load_dotenv(dotenv_path=self.env_file_path, override=True)

        env_var_name_attempt = self._generate_env_var_name(public_address, entity_id_hint)
        key_found_and_unset = False
        if os.getenv(env_var_name_attempt) is not None:
            if unset_key(self.env_file_path, env_var_name_attempt):
                logger.info(f"{self.log_prefix} Deprecated (unset) key '{env_var_name_attempt}' for {public_address}.")
                key_found_and_unset = True
            else: # pragma: no cover
                logger.error(f"{self.log_prefix} Failed to unset key '{env_var_name_attempt}' using dotenv.unset_key.")
        
        if not key_found_and_unset and entity_id_hint is not None: # Try fallback name
            env_var_name_fallback = self._generate_env_var_name(public_address, None)
            if os.getenv(env_var_name_fallback) is not None:
                if unset_key(self.env_file_path, env_var_name_fallback): # pragma: no cover
                    logger.info(f"{self.log_prefix} Deprecated (unset) fallback key '{env_var_name_fallback}' for {public_address}.")
                    key_found_and_unset = True
                else: # pragma: no cover
                    logger.error(f"{self.log_prefix} Failed to unset fallback key '{env_var_name_fallback}'.")

        if not key_found_and_unset:
            logger.warning(f"{self.log_prefix} No key found to deprecate for {public_address} with hint '{entity_id_hint}'.")
            return False
        return True


    def list_managed_identities(self) -> List[Dict[str, str]]: # pragma: no cover
        """
        Lists identities by parsing variable names from the agent's .env file.
        Returns a list of dicts, each with "entity_id_part", "public_address_suffix", "env_var_name".
        """
        identities = []
        if not self.env_file_path.exists(): return identities
        
        # Reading the .env file directly to parse keys, as os.environ might be polluted
        # or not reflect file changes if not reloaded by this process.
        # python-dotenv's dotenv_values gives a dict of the file content.
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(self.env_file_path)
        except Exception as e: # pragma: no cover
            logger.error(f"{self.log_prefix} Could not read .env file {self.env_file_path} for listing: {e}")
            return []

        prefix = "MINDX_WALLET_PK_"
        for key in env_values.keys():
            if key.startswith(prefix):
                try:
                    parts_str = key[len(prefix):] # <ENTITY_ID_SLUG>_<PUBLIC_ADDRESS_UPPER>
                    # This parsing is fragile if entity_id itself contains underscores.
                    # Assuming format MINDX_WALLET_PK_ENTITY_SUFFIX
                    # Or MINDX_WALLET_PK_ENTITY_ADDRSUFFIX
                    # Let's assume the suffix is always the address part for now if it matches 0x format
                    
                    # Try to find a 42-char hex string (0x + 40 hex) in the latter part of parts_str
                    addr_match = re.search(r"(0X[0-9A-F]{40})", parts_str.upper())
                    if addr_match:
                        public_address = addr_match.group(1)
                        entity_part = parts_str.replace(f"_{public_address}","").replace(public_address,"") # Basic attempt to get entity
                        identities.append({
                            "entity_id_part": entity_part.replace("_UNSPECIFIED", "") if entity_part != "_UNSPECIFIED" else "general_purpose",
                            "public_address": public_address, # Store full address if parsed
                            "env_var_name": key
                        })
                    else: # Fallback parsing if no clear 0xADDRESS found
                        split_parts = parts_str.split('_')
                        public_address_suffix = split_parts[-1] if len(split_parts) > 1 else "UNKNOWN"
                        entity_id_part = "_".join(split_parts[:-1]) if len(split_parts) > 1 else parts_str
                        identities.append({
                            "entity_id_part": entity_id_part if entity_id_part else "unknown_entity",
                            "public_address_suffix": public_address_suffix,
                            "env_var_name": key
                        })
                except Exception as e_parse: # pragma: no cover
                    logger.warning(f"{self.log_prefix} Could not parse env var name '{key}': {e_parse}")
        
        logger.info(f"{self.log_prefix} Found {len(identities)} potential identities in {self.env_file_path}.")
        return identities
        
    async def shutdown(self): # pragma: no cover
        """Perform any cleanup for the IDManagerAgent."""
        logger.info(f"IDManagerAgent '{self.agent_id}' shutting down. (Conceptual: ensure .env file is secured if using Tomb-like system).")
        # If using Tomb/EmbalmerAgent, this is where self.embalmer.close_tomb() would be called.
    
    @classmethod
    async def reset_all_instances_for_testing(cls): # pragma: no cover
        """Resets all cached named instances. Primarily for testing."""
        async with cls._class_lock:
            for agent_id, instance in list(cls._instances.items()):
                if hasattr(instance, 'shutdown') and asyncio.iscoroutinefunction(instance.shutdown):
                    await instance.shutdown()
            cls._instances.clear()
        logger.debug("All IDManagerAgent instances reset.")
