#!/bin/bash

# AGENTIC_MINDX_DEPLOY_V2.0.0
# Production-focused setup and (conceptual) service preparation for MindX.
# This script aims to be robust and configurable for deploying the MindX system.

# Strict mode
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Pipeline's return status is the last non-zero command.
# set -u # Treat unset variables as an error (can be too strict).

# --- Script Version & Defaults ---
SCRIPT_VERSION="2.0.0"
DEFAULT_PROJECT_ROOT_NAME="augmentic_mindx" # Used if deploying into a new dir
DEFAULT_VENV_NAME=".mindx_env" # Changed for clarity
DEFAULT_FRONTEND_PORT="3000"
DEFAULT_BACKEND_PORT="8000"
DEFAULT_LOG_LEVEL="INFO" # For MindX application

# --- Logging ---
# Script's own log file
SETUP_LOG_FILE="" # Will be set after PROJECT_ROOT is confirmed

function log_setup_info { echo "[SETUP INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$SETUP_LOG_FILE"; echo "[INFO] $1"; }
function log_setup_warn { echo "[SETUP WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$SETUP_LOG_FILE"; echo "[WARN] $1"; }
function log_setup_error { echo "[SETUP ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$SETUP_LOG_FILE"; echo "[ERROR] $1" >&2; }

# --- Initial Configuration & Argument Parsing ---
# This script can be run from anywhere. It will create/use PROJECT_ROOT.
# If PROJECT_ROOT is not specified, it defaults to a subdirectory in current dir or a specified path.

TARGET_INSTALL_DIR_ARG="" # Path where augmentic_mindx project will reside or be created
MINDX_CONFIG_FILE_ARG="" # Optional path to a pre-existing mindx_config.json
DOTENV_FILE_ARG=""      # Optional path to a pre-existing .env file
RUN_SERVICES_FLAG=false
INTERACTIVE_SETUP_FLAG=false

function show_help { # pragma: no cover
    echo "MindX Deployment Script v${SCRIPT_VERSION}"
    echo "Usage: $0 [options] <target_install_directory>"
    echo ""
    echo "Arguments:"
    echo "  <target_install_directory>   The root directory where the '${DEFAULT_PROJECT_ROOT_NAME}' project will be located or created."
    echo "                               If it exists and contains MindX, it will be configured. If not, MindX structure will be created."
    echo ""
    echo "Options:"
    echo "  --config-file <path>         Path to an existing mindx_config.json to use."
    echo "  --dotenv-file <path>         Path to an existing .env file to copy into the project."
    echo "  --run                        Start MindX backend and frontend services after setup."
    echo "  --interactive                Prompt for key configuration values if not found in .env or args."
    echo "  --venv-name <name>           Override default virtual environment name (Default: ${DEFAULT_VENV_NAME})."
    echo "  --frontend-port <port>       Override default frontend port (Default: ${DEFAULT_FRONTEND_PORT})."
    echo "  --backend-port <port>        Override default backend port (Default: ${DEFAULT_BACKEND_PORT})."
    echo "  --log-level <level>          MindX application log level (DEBUG, INFO, etc. Default: ${DEFAULT_LOG_LEVEL})."
    echo "  -h, --help                   Show this help message."
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-file) MINDX_CONFIG_FILE_ARG="$2"; shift 2;;
        --dotenv-file) DOTENV_FILE_ARG="$2"; shift 2;;
        --run) RUN_SERVICES_FLAG=true; shift 1;;
        --interactive) INTERACTIVE_SETUP_FLAG=true; shift 1;;
        --venv-name) MINDX_VENV_NAME_OVERRIDE="$2"; shift 2;;
        --frontend-port) FRONTEND_PORT_OVERRIDE="$2"; shift 2;;
        --backend-port) BACKEND_PORT_OVERRIDE="$2"; shift 2;;
        --log-level) LOG_LEVEL_OVERRIDE="$2"; shift 2;;
        -h|--help) show_help;;
        *)
            if [[ -z "$TARGET_INSTALL_DIR_ARG" ]] && [[ ! "$1" =~ ^-- ]]; then
                TARGET_INSTALL_DIR_ARG="$1"
                shift 1
            else
                log_setup_error "Unknown option or too many arguments: $1"
                show_help # Will exit
            fi
            ;;
    esac
done

if [[ -z "$TARGET_INSTALL_DIR_ARG" ]]; then
    log_setup_error "Target installation directory not specified."
    show_help # Will exit
fi

# Resolve and create project root
PROJECT_ROOT=$(readlink -f "$TARGET_INSTALL_DIR_ARG") # Get absolute path
if [ ! -d "$PROJECT_ROOT/$DEFAULT_PROJECT_ROOT_NAME" ]; then
    # If augmentic_mindx doesn't exist inside target, assume target IS project root.
    # Or, if target is meant to *contain* augmentic_mindx, create it.
    # For this script, let's assume TARGET_INSTALL_DIR_ARG *is* the project root.
    log_info "Target directory '$PROJECT_ROOT' will be used as MindX project root."
    mkdir -p "$PROJECT_ROOT" || { log_setup_error "Failed to create project root: $PROJECT_ROOT"; exit 1; }
else
    PROJECT_ROOT="$PROJECT_ROOT/$DEFAULT_PROJECT_ROOT_NAME" # If target dir contains it
    log_info "MindX project found within target directory at: $PROJECT_ROOT"
fi


# --- Setup Script Log File (now that PROJECT_ROOT is confirmed) ---
mkdir -p "$PROJECT_ROOT/data/logs" || { echo "[ERROR] Critical: Failed to create $PROJECT_ROOT/data/logs for setup log."; exit 1; }
SETUP_LOG_FILE="$PROJECT_ROOT/data/logs/mindx_deployment_setup.log"
# Redirect all stdout/stderr of this script to a tee command to capture in file and show on console
# This is complex to do for the whole script after it has started.
# Simpler: Ensure all log_setup_* functions write to file. We'll do that.
echo "--- MindX Deployment Log $(date) ---" > "$SETUP_LOG_FILE" # Initialize/clear log

log_info "MindX Deployment Script v${SCRIPT_VERSION}"
log_info "Final Project Root: $PROJECT_ROOT"

# --- Override Defaults with CLI Args / Env Vars ---
MINDX_VENV_NAME="${MINDX_VENV_NAME_OVERRIDE:-${MINDX_VENV_NAME:-$DEFAULT_VENV_NAME}}"
FRONTEND_PORT_EFFECTIVE="${FRONTEND_PORT_OVERRIDE:-${FRONTEND_PORT:-$DEFAULT_FRONTEND_PORT}}"
BACKEND_PORT_EFFECTIVE="${BACKEND_PORT_OVERRIDE:-${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}}"
MINDX_APP_LOG_LEVEL="${LOG_LEVEL_OVERRIDE:-${MINDX_LOG_LEVEL:-$DEFAULT_LOG_LEVEL}}" # For app, not this script

# --- Derived Paths (Absolute) ---
MINDX_VENV_PATH_ABS="$PROJECT_ROOT/$MINDX_VENV_NAME"
MINDX_BACKEND_SERVICE_DIR_ABS="$PROJECT_ROOT/mindx_backend_service" # Keep service separate from 'mindx' package
MINDX_FRONTEND_UI_DIR_ABS="$PROJECT_ROOT/mindx_frontend_ui"
MINDX_DATA_DIR_ABS="$PROJECT_ROOT/data"
MINDX_LOGS_DIR_ABS="$MINDX_DATA_DIR_ABS/logs" # For application logs
MINDX_PIDS_DIR_ABS="$MINDX_DATA_DIR_ABS/pids"
MINDX_CONFIG_DIR_ABS="$MINDX_DATA_DIR_ABS/config"


MINDX_BACKEND_APP_LOG_FILE="$MINDX_LOGS_DIR_ABS/mindx_coordinator_service.log"
MINDX_FRONTEND_APP_LOG_FILE="$MINDX_LOGS_DIR_ABS/mindx_frontend_service.log"
BACKEND_PID_FILE="$MINDX_PIDS_DIR_ABS/mindx_backend.pid"
FRONTEND_PID_FILE="$MINDX_PIDS_DIR_ABS/mindx_frontend.pid"

# --- Helper Functions (Continued) ---
function check_command_presence { # pragma: no cover
  if ! command -v "$1" &> /dev/null; then
    log_setup_error "'$1' is not installed or not in PATH. Please install it."
    # Add more specific help for missing commands if possible
    exit 1
  fi
}

function create_or_overwrite_file_secure { # pragma: no cover
    local file_path_abs="$1"; local content="$2"; local perms="${3:-600}" # Default to 600 for sensitive files
    local dir_path_abs; dir_path_abs=$(dirname "$file_path_abs")
    if ! mkdir -p "$dir_path_abs"; then log_setup_error "Failed to create directory: $dir_path_abs"; exit 1; fi
    log_setup_info "Creating/Overwriting file: $file_path_abs with permissions $perms"
    # Write content first
    if [ -n "$content" ]; then
        if ! printf '%s\n' "$content" > "$file_path_abs"; then log_setup_error "Failed write to: $file_path_abs"; exit 1; fi
    else
        if ! > "$file_path_abs"; then log_setup_error "Failed create empty: $file_path_abs"; exit 1; fi
    fi
    # Set permissions
    if ! chmod "$perms" "$file_path_abs"; then log_setup_warning "Failed to set permissions $perms for $file_path_abs"; fi
}

function check_python_venv_command { # pragma: no cover
  # Assumes venv is active when called
  local cmd_to_check="$1"; local install_package_name="$2"
  install_package_name="${install_package_name:-$cmd_to_check}"
  if ! command -v "$cmd_to_check" &> /dev/null; then
    log_setup_warn "'$cmd_to_check' not found in venv. Attempting install: '$install_package_name'..."
    if ! python -m pip install "$install_package_name" -q; then log_setup_error "Failed to install '$install_package_name'."; return 1; fi
    log_setup_info "'$install_package_name' installed in venv."
    if ! command -v "$cmd_to_check" &> /dev/null; then log_setup_error "'$cmd_to_check' still not found."; return 1; fi
  fi
  log_setup_info "'$cmd_to_check' confirmed in venv."
  return 0
}

# --- Create Base Project Structure if not exists ---
function ensure_mindx_structure {
    log_setup_info "Ensuring MindX base directory structure exists at $PROJECT_ROOT..."
    mkdir -p "$PROJECT_ROOT/mindx/core"
    mkdir -p "$PROJECT_ROOT/mindx/orchestration"
    mkdir -p "$PROJECT_ROOT/mindx/learning"
    mkdir -p "$PROJECT_ROOT/mindx/monitoring"
    mkdir -p "$PROJECT_ROOT/mindx/llm"
    mkdir -p "$PROJECT_ROOT/mindx/utils"
    mkdir -p "$PROJECT_ROOT/mindx/tools" # For BaseGenAgent etc.
    mkdir -p "$PROJECT_ROOT/mindx/docs"   # Stub dir
    mkdir -p "$PROJECT_ROOT/scripts"
    mkdir -p "$MINDX_DATA_DIR_ABS" # Central data directory
    mkdir -p "$MINDX_LOGS_DIR_ABS"
    mkdir -p "$MINDX_PIDS_DIR_ABS"
    mkdir -p "$MINDX_CONFIG_DIR_ABS" # For basegen_config.json etc.

    # Create minimal __init__.py files to make them packages
    find "$PROJECT_ROOT/mindx" -type d -exec touch {}/__init__.py \;
    touch "$PROJECT_ROOT/scripts/__init__.py" # If scripts are also to be importable

    # Check if core MindX source files (e.g. coordinator, sia) exist.
    # If not, this script cannot proceed with actually *running* MindX.
    # For a true "installer", it would fetch/copy these files.
    # For this script, we assume they are already part of the $PROJECT_ROOT (e.g. git cloned).
    if [ ! -f "$PROJECT_ROOT/mindx/orchestration/coordinator_agent.py" ] || \
       [ ! -f "$PROJECT_ROOT/mindx/learning/self_improve_agent.py" ]; then
        log_setup_warn "Core MindX agent source files (coordinator_agent.py, self_improve_agent.py) not found in $PROJECT_ROOT/mindx/..."
        log_setup_warn "This script primarily sets up the environment and services for an EXISTING MindX codebase."
        log_setup_warn "If you intended to deploy MindX code, ensure it's present in $PROJECT_ROOT first."
        # Optionally, exit here if code is mandatory:
        # log_setup_error "MindX source code not found. Deployment cannot continue."
        # exit 1
    fi
    log_setup_info "MindX base directory structure ensured."
}


# --- Configuration File Management ---
function setup_dotenv_file {
    local target_dotenv_path="$PROJECT_ROOT/.env"
    if [ -f "$target_dotenv_path" ] && [ -z "$DOTENV_FILE_ARG" ]; then
        log_setup_info ".env file already exists at $target_dotenv_path. Skipping creation unless --dotenv-file is used to overwrite."
        return 0
    fi

    local env_content_source_path=""
    if [ -n "$DOTENV_FILE_ARG" ]; then
        if [ -f "$DOTENV_FILE_ARG" ]; then
            env_content_source_path="$DOTENV_FILE_ARG"
            log_setup_info "Using provided .env file: $env_content_source_path"
        else
            log_setup_error "Provided --dotenv-file '$DOTENV_FILE_ARG' not found. Using default content."
        fi
    fi

    if [ -n "$env_content_source_path" ]; then
        cp "$env_content_source_path" "$target_dotenv_path" || { log_setup_error "Failed to copy provided .env file."; exit 1; }
        chmod 600 "$target_dotenv_path" # Secure permissions for .env
    else
        log_setup_info "No .env file provided or found. Creating a default .env at $target_dotenv_path"
        # Prompt for Gemini API Key if interactive and not set
        local gemini_api_key_val=""
        if [[ "$INTERACTIVE_SETUP_FLAG" == true ]]; then
            read -r -p "Enter your Google Gemini API Key (or press Enter to skip): " gemini_api_key_val
        fi
        gemini_api_key_val="${gemini_api_key_val:-YOUR_GEMINI_API_KEY_HERE}"


        # Default .env content
        read -r -d '' default_env_content <<EOF_DEFAULT_DOTENV
# MindX System Environment Configuration (.env)
# This file is loaded by mindx.utils.Config

# --- General Logging Level ---
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
MINDX_LOG_LEVEL="${MINDX_APP_LOG_LEVEL}"

# --- Default LLM Provider for the whole system ---
# Can be overridden by specific agent configurations.
# Options: "ollama", "gemini" (add more in mindx/llm/llm_factory.py)
MINDX_LLM__DEFAULT_PROVIDER="ollama"

# --- Ollama Specific Configuration ---
MINDX_LLM__OLLAMA__DEFAULT_MODEL="nous-hermes2:latest"
MINDX_LLM__OLLAMA__DEFAULT_MODEL_FOR_CODING="deepseek-coder:6.7b-instruct"
MINDX_LLM__OLLAMA__DEFAULT_MODEL_FOR_REASONING="nous-hermes2:latest"
# MINDX_LLM__OLLAMA__BASE_URL="http://localhost:11434" # Default, uncomment to override

# --- Gemini Specific Configuration ---
# IMPORTANT: Get your API key from Google AI Studio (https://aistudio.google.com/app/apikey)
# This key will be used if MINDX_LLM__GEMINI__API_KEY is not set directly below.
GEMINI_API_KEY="${gemini_api_key_val}"
MINDX_LLM__GEMINI__API_KEY="${gemini_api_key_val}"
MINDX_LLM__GEMINI__DEFAULT_MODEL="gemini-1.5-flash-latest"
MINDX_LLM__GEMINI__DEFAULT_MODEL_FOR_CODING="gemini-1.5-pro-latest" # Or flash for cost/speed
MINDX_LLM__GEMINI__DEFAULT_MODEL_FOR_REASONING="gemini-1.5-pro-latest"


# --- SelfImprovementAgent (SIA) Specific LLM Configuration ---
MINDX_SELF_IMPROVEMENT_AGENT__LLM__PROVIDER="ollama"
MINDX_SELF_IMPROVEMENT_AGENT__LLM__MODEL="deepseek-coder:6.7b-instruct" # Needs to be good at Python

# --- CoordinatorAgent Specific LLM Configuration ---
MINDX_COORDINATOR__LLM__PROVIDER="ollama"
MINDX_COORDINATOR__LLM__MODEL="nous-hermes2:latest" # For system analysis, etc.

# --- Coordinator's Autonomous Improvement Loop ---
MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__ENABLED="false" # Recommended: false for initial setup
MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__INTERVAL_SECONDS="3600" # 1 hour
MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__REQUIRE_HUMAN_APPROVAL_FOR_CRITICAL="true"
# Critical components list is managed in mindx_config.json or code defaults

# --- Monitoring ---
MINDX_MONITORING__RESOURCE__ENABLED="true"
MINDX_MONITORING__RESOURCE__INTERVAL="15.0" # Check resources every 15 seconds
MINDX_MONITORING__PERFORMANCE__ENABLE_PERIODIC_SAVE="true"
MINDX_MONITORING__PERFORMANCE__PERIODIC_SAVE_INTERVAL_SECONDS="300" # Save perf metrics every 5 mins

# --- Backend Service (FastAPI) ---
# MINDX_BACKEND_SERVICE__ALLOW_ALL_ORIGINS="false" # Set to "true" for dev if needed for CORS, careful in prod.

# --- Ports (used by this script if not overridden by CLI args or shell env) ---
# FRONTEND_PORT="${FRONTEND_PORT_EFFECTIVE}" # Set by script variables
# BACKEND_PORT="${BACKEND_PORT_EFFECTIVE}"  # Set by script variables
EOF_DEFAULT_DOTENV
        create_or_overwrite_file_secure "$target_dotenv_path" "$default_env_content" "600" # Restrictive perms
    fi
    log_setup_info ".env file configured at $target_dotenv_path"
}

function setup_mindx_config_json { # pragma: no cover
    local target_config_path="$MINDX_CONFIG_DIR_ABS/mindx_config.json" # Centralized location
    if [ -f "$target_config_path" ] && [ -z "$MINDX_CONFIG_FILE_ARG" ]; then
        log_setup_info "mindx_config.json already exists at $target_config_path. Skipping."
        return 0
    fi

    if [ -n "$MINDX_CONFIG_FILE_ARG" ]; then
        if [ -f "$MINDX_CONFIG_FILE_ARG" ]; then
            cp "$MINDX_CONFIG_FILE_ARG" "$target_config_path" || { log_setup_error "Failed to copy provided mindx_config.json."; exit 1; }
            log_setup_info "Used provided mindx_config.json: $MINDX_CONFIG_FILE_ARG"
        else
            log_setup_error "Provided --config-file '$MINDX_CONFIG_FILE_ARG' not found. Using default content."
            # Fall through to create default
        fi
    fi
    
    if [ ! -f "$target_config_path" ]; then # Create default if still not present
        log_setup_info "Creating default mindx_config.json at $target_config_path"
        # This default JSON complements .env; .env/environment vars will override these.
        read -r -d '' default_json_config_content << 'EOF_DEFAULT_JSON_CONFIG'
{
  "system": {
    "version": "0.4.0",
    "name": "MindX Self-Improving System (Augmentic)"
  },
  "logging": {
    "uvicorn_level": "info" 
  },
  "llm": {
    "providers": {
      "ollama": {"enabled": true},
      "gemini": {"enabled": true}
    }
  },
  "self_improvement_agent": {
    "analysis": {
      "max_code_chars": 70000,
      "max_description_tokens": 350
    },
    "implementation": {
      "max_code_gen_tokens": 12000,
      "temperature": 0.05
    },
    "evaluation": {
      "max_chars_for_critique": 4000,
      "max_critique_tokens": 300
    }
  },
  "coordinator": {
    "autonomous_improvement": {
      "critical_components": [
        "mindx.learning.self_improve_agent",
        "mindx.orchestration.coordinator_agent",
        "mindx.utils.config"
      ]
    }
  },
  "tools": {
    "note_taking": {
        "enabled": true,
        "notes_dir_relative_to_project": "data/bdi_agent_notes" 
    },
    "summarization": {
        "enabled": true,
        "max_input_chars": 30000
        # LLM for summarization tool can be configured here too, e.g.
        # "llm": {"provider": "ollama", "model": "mistral"}
    },
    "web_search": {
        "enabled": true, # Requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env
        "timeout_seconds": 20.0
        # API keys should be in .env, not here
    }
  }
}
EOF_DEFAULT_JSON_CONFIG
        create_or_overwrite_file_secure "$target_config_path" "$default_json_config_content" "644" # Readable by all
    fi
    log_setup_info "mindx_config.json configured at $target_config_path"
}


# --- Python Virtual Environment Setup (Simplified - Main one done by setup_mindx_deps) ---
# Individual service venv setup removed as main project venv is now primary.

# --- Backend Service Setup ---
# (setup_backend_service from previous version - mostly file creation)
# (It assumes Python dependencies are already handled by the main venv)
function setup_backend_service { # pragma: no cover
  log_info "Setting up MindX Backend Service files in '$MINDX_BACKEND_SERVICE_DIR_ABS'..."
  mkdir -p "$MINDX_BACKEND_SERVICE_DIR_ABS"
  # FastAPI main application (main_service.py)
  # Content is the FULL main_service.py from previous response (the one with API endpoints)
  # For brevity, I'm using a placeholder here. In the actual script, paste the full content.
  local backend_main_py_content; backend_main_py_content=$(cat "$PROJECT_ROOT/TEMPLATES/backend_main_service.py.template") # Conceptual: load from template
  # **IMPORTANT**: Replace the line above with the actual heredoc or cat of the full backend_main_service.py content
  # from the previous response if you don't have a template file. Example:
  # read -r -d '' backend_main_py_content << 'EOF_BACKEND_MAIN_SERVICE_TEMPLATE_CONTENT'
  # # ... (Full backend_main_service.py content from previous response here) ...
  # EOF_BACKEND_MAIN_SERVICE_TEMPLATE_CONTENT
  # For this script to be self-contained without external templates:
  # THIS IS A CRITICAL PLACEHOLDER. The actual FastAPI app code is complex.
  # Refer to the previous response for the full main_service.py code.
  read -r -d '' backend_main_py_content << 'EOF_PLACEHOLDER_BACKEND'
import uvicorn, os
from fastapi import FastAPI
app = FastAPI(title="MindX Backend Placeholder")
@app.get("/api/health")
async def health(): return {"status": "ok", "message": "MindX Backend Placeholder Running"}
if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8000")), log_level="info")
EOF_PLACEHOLDER_BACKEND
  create_or_overwrite_file "$MINDX_BACKEND_SERVICE_DIR_ABS/main_service.py" "$backend_main_py_content"
  log_info "MindX Backend Service main_service.py created (using placeholder/template)."
}


# --- Frontend UI Setup ---
# (setup_frontend_ui from previous version - file creation)
# (It assumes Node.js/npm dependencies will be installed if needed)
function setup_frontend_ui { # pragma: no cover
  log_info "Setting up MindX Frontend UI files in '$MINDX_FRONTEND_UI_DIR_ABS'..."
  mkdir -p "$MINDX_FRONTEND_UI_DIR_ABS"
  # index.html, styles.css, app.js, package.json, server_static.js
  # For brevity, these are placeholders. Use full content from previous response.
  create_or_overwrite_file "$MINDX_FRONTEND_UI_DIR_ABS/index.html" "<!DOCTYPE html><html><head><title>MindX UI Placeholder</title></head><body><h1>MindX UI Placeholder</h1><p>See app.js and full script for actual UI.</p><script>window.MINDX_BACKEND_PORT=\"${BACKEND_PORT_EFFECTIVE}\";</script><script src=\"app.js\"></script></body></html>"
  create_or_overwrite_file "$MINDX_FRONTEND_UI_DIR_ABS/styles.css" "body { font-family: sans-serif; padding: 20px; }"
  create_or_overwrite_file "$MINDX_FRONTEND_UI_DIR_ABS/app.js" "console.log('MindX Frontend UI Placeholder JS. Backend expected on port: ' + window.MINDX_BACKEND_PORT);"
  create_or_overwrite_file "$MINDX_FRONTEND_UI_DIR_ABS/package.json" "{ \"name\": \"mindx-frontend-placeholder\", \"version\": \"0.1.0\", \"main\": \"server_static.js\", \"scripts\": {\"start\":\"node server_static.js\"}, \"dependencies\":{\"express\":\"^4.18.0\"} }"
  create_or_overwrite_file "$MINDX_FRONTEND_UI_DIR_ABS/server_static.js" "const express=require('express'); const path=require('path'); const app=express(); const port=process.env.FRONTEND_PORT||3000; app.use(express.static(__dirname)); app.get('*',(req,res)=>res.sendFile(path.resolve(__dirname,'index.html'))); app.listen(port,'0.0.0.0',()=>{console.log(\`MindX FE Placeholder on http://localhost:\${port}\`)});"
  
  log_info "MindX Frontend UI files created (using placeholders). Installing Express..."
  local current_dir_for_npm; current_dir_for_npm=$(pwd)
  cd "$MINDX_FRONTEND_UI_DIR_ABS" || { log_setup_error "Failed to cd to frontend dir for npm install."; return 1; }
  if [ -f "package.json" ]; then
    log_info "Running 'npm install' for frontend..."
    # In production, consider `npm ci` if package-lock.json is committed
    npm install --silent >> "$MINDX_FRONTEND_APP_LOG_FILE" 2>&1 || { log_setup_error "npm install failed for frontend. Check $MINDX_FRONTEND_APP_LOG_FILE."; cd "$current_dir_for_npm" || exit 1; return 1; }
    log_info "Frontend dependencies installed."
  fi
  cd "$current_dir_for_npm" || { log_setup_error "Failed to cd back after npm install."; exit 1; }
  return 0
}

# --- Service Start/Stop Functions (Conceptual for systemd/supervisor) ---
# These functions would generate service unit files or supervisor config files.
# For direct backgrounding, we use simpler start/stop.

function start_mindx_service { # pragma: no cover
    local service_name="$1" # "backend" or "frontend"
    local exec_command="$2"
    local pid_file="$3"
    local log_file="$4"
    local service_dir="$5" # Directory to cd into before running

    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null; then
        log_setup_info "Service '$service_name' already running (PID: $(cat "$pid_file"))."
        return 0
    fi

    log_setup_info "Starting MindX Service '$service_name'..."
    mkdir -p "$(dirname "$pid_file")" "$(dirname "$log_file")" # Ensure dirs

    local current_dir_svc_start; current_dir_svc_start=$(pwd)
    cd "$service_dir" || { log_setup_error "Failed to cd to $service_dir for $service_name"; return 1; }

    # Use nohup to detach, redirect output, and run in background
    # The command needs to be constructed carefully.
    # Example for backend: nohup $venv_python main_service.py >> $log_file 2>&1 &
    # Example for frontend: nohup node server_static.js >> $log_file 2>&1 &
    nohup $exec_command >> "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    
    cd "$current_dir_svc_start" || exit 1 # Should not fail

    log_setup_info "MindX Service '$service_name' process initiated with PID: $pid."
    log_setup_info "Allowing a few seconds for '$service_name' to stabilize..."
    sleep 5 
    if ! ps -p "$pid" > /dev/null; then # pragma: no cover
        log_setup_error "Service '$service_name' (PID $pid) failed to stay running. Check logs: $log_file"
        rm -f "$pid_file"
        return 1
    fi
    log_setup_info "Service '$service_name' appears to be running."
    return 0
}

function stop_mindx_service { # pragma: no cover
    local pid_file="$1"; local service_name="$2"
    # (Same robust stop_service logic from previous script version)
    if [ -f "$pid_file" ]; then
        local pid_val; pid_val=$(cat "$pid_file");
        if [ -n "$pid_val" ] && ps -p "$pid_val" > /dev/null; then
            log_setup_info "Stopping $service_name (PID: $pid_val)..."; kill -TERM "$pid_val" &>/dev/null; sleep 2;
            if ps -p "$pid_val" > /dev/null; then log_setup_warn "$service_name (PID $pid_val) TERM fail, sending KILL..."; kill -KILL "$pid_val" &>/dev/null; sleep 1; fi
            if ps -p "$pid_val" > /dev/null; then log_setup_error "$service_name (PID $pid_val) FAILED TO STOP."
            else log_setup_info "$service_name stopped."
            fi
        else log_setup_info "$service_name (PID: $pid_val from file) already stopped or PID invalid."
        fi; rm -f "$pid_file" # Clean up PID file
    else log_setup_info "$service_name PID file '$pid_file' not found (might be first run or already cleaned)."
    fi
}

# --- Cleanup Function for traps ---
function cleanup_on_exit_final { # pragma: no cover
    log_setup_info "--- MindX Deployment Script Exiting: Initiating Final Cleanup ---"
    stop_mindx_service "$FRONTEND_PID_FILE" "MindX Frontend UI"
    stop_mindx_service "$BACKEND_PID_FILE" "MindX Backend Service"
    
    # Deactivate venv only if this script sourced it and it's still active.
    # This can be unreliable in traps. Best effort.
    if [[ -n "$VIRTUAL_ENV" ]] && [[ "$VIRTUAL_ENV" == "$MINDX_VENV_PATH_ABS" ]]; then
        log_setup_info "Attempting to deactivate MindX virtual environment from trap..."
        deactivate || log_setup_warn "Deactivate command failed or venv not active in this trap's shell context."
    fi
    log_setup_info "Cleanup attempt complete. MindX services signaled to stop."
    log_setup_info ">>> Augmentic MindX Deployment Script Terminated <<<"
    # Do not exit here if called by trap on EXIT, it causes recursion.
    # If called by SIGINT/SIGTERM, exit 0 makes sure the script exits cleanly after trap.
    if [[ "$_TRAP_SIGNAL" != "EXIT" ]]; then
        exit 0
    fi
}
# Store signal for trap handler
_TRAP_SIGNAL=""
function trap_handler_proxy { # pragma: no cover
    _TRAP_SIGNAL="$1" # Store the signal name
    cleanup_on_exit_final # Call the actual cleanup
}
trap 'trap_handler_proxy "SIGINT"' SIGINT
trap 'trap_handler_proxy "SIGTERM"' SIGTERM
trap 'trap_handler_proxy "EXIT"' EXIT


# --- Main Execution Flow ---
log_setup_info ">>> Starting Augmentic MindX Deployment Script (v${SCRIPT_VERSION}) <<<"
log_setup_info "Target Project Root: $PROJECT_ROOT"
log_setup_info "Application Log Level will be set to: $MINDX_APP_LOG_LEVEL"

# Create base directories
ensure_mindx_structure # Creates mindx package structure, data, logs, pids, config dirs

# Configure .env and mindx_config.json
setup_dotenv_file # Handles .env based on args or defaults
setup_mindx_config_json # Handles mindx_config.json

# Setup Python Environment
setup_virtual_environment_and_mindx_deps || { log_setup_error "Python environment setup failed. Exiting."; exit 1; }
# Venv is kept active for now. Service start functions will manage their own context.

# Setup Backend and Frontend files (code generation within this script)
setup_backend_service || { log_setup_error "Backend Service file setup failed. Exiting."; deactivate_venv_if_active; exit 1; }
setup_frontend_ui || { log_setup_error "Frontend UI file setup failed. Exiting."; deactivate_venv_if_active; exit 1; }

# Deactivate main script's venv sourcing if any, services run in their own context
function deactivate_venv_if_active { # Helper for cleanup before exit
    if [[ -n "$VIRTUAL_ENV" ]] && [[ "$VIRTUAL_ENV" == "$MINDX_VENV_PATH_ABS" ]]; then
        log_setup_info "Deactivating main script's venv sourcing..."
        deactivate || log_setup_warn "Deactivate command failed in main script scope."
    fi
}
deactivate_venv_if_active

if [[ "$RUN_SERVICES_FLAG" == true ]]; then
    log_setup_info "--- Starting MindX Services (Backend & Frontend) ---"
    
    # Backend command construction
    # The backend's main_service.py now calls uvicorn.run itself.
    # We need to ensure it uses the venv's python.
    VENV_PYTHON="$MINDX_VENV_PATH_ABS/bin/python"
    BACKEND_EXEC_COMMAND="$VENV_PYTHON $MINDX_BACKEND_SERVICE_DIR_ABS/main_service.py"

    start_mindx_service "MindX Backend Service" "$BACKEND_EXEC_COMMAND" "$BACKEND_PID_FILE" "$MINDX_BACKEND_APP_LOG_FILE" "$MINDX_BACKEND_SERVICE_DIR_ABS" || \
        { log_setup_error "MindX Backend Service failed to start. Check logs. Exiting."; exit 1; }

    # Frontend command construction
    # Node doesn't need venv, but ensure node and server_static.js are found
    NODE_EXEC_COMMAND="node server_static.js" # server_static.js is in MINDX_FRONTEND_UI_DIR_ABS
    start_mindx_service "MindX Frontend UI" "$NODE_EXEC_COMMAND" "$FRONTEND_PID_FILE" "$MINDX_FRONTEND_APP_LOG_FILE" "$MINDX_FRONTEND_UI_DIR_ABS" || \
        { log_setup_error "MindX Frontend UI failed to start. Check logs. Exiting."; exit 1; }

    log_setup_info ">>> Augmentic MindX System Services Started <<<"
    log_setup_info "  Backend API: http://localhost:$BACKEND_PORT_EFFECTIVE (and on 0.0.0.0)"
    log_setup_info "  Backend Logs: tail -f $MINDX_BACKEND_APP_LOG_FILE"
    log_setup_info "  Frontend UI: http://localhost:$FRONTEND_PORT_EFFECTIVE"
    log_setup_info "  Frontend Logs: tail -f $MINDX_FRONTEND_APP_LOG_FILE"
    log_setup_info ">>> Press Ctrl+C to stop all services and exit this script. <<<"

    log_setup_info "Deployment script running in foreground, monitoring services via 'wait'."
    wait # Wait for background jobs started by this script's shell
    log_setup_info "Deployment script 'wait' command finished or interrupted."
else
    log_setup_info "MindX setup complete. Services not started due to --run flag not being set."
    log_setup_info "To start services manually (example):"
    log_setup_info "  Backend: cd $MINDX_BACKEND_SERVICE_DIR_ABS && $MINDX_VENV_PATH_ABS/bin/python main_service.py"
    log_setup_info "  Frontend: cd $MINDX_FRONTEND_UI_DIR_ABS && node server_static.js"
fi

# Cleanup will be called by the EXIT trap automatically.
# No explicit call to cleanup_on_exit_final needed here.
