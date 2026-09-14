"""
Configuration loading utilities with OmegaConf support.

This module provides a unified way to load YAML configuration files with:
- Variable interpolation (e.g., ${paths.datasets.mvtec})
- Environment variable support (e.g., ${oc.env:FASTAD_ROOT,/default/path})
- Automatic merging with base paths.yaml

Usage:
    from config_utils import load_config
    config = load_config("configs/MVTec_AD/cable/base.yaml")
"""

import os
from typing import Union
from omegaconf import OmegaConf, DictConfig


def find_project_root(start_path: str = None) -> str:
    """Find the project root directory by looking for configs/paths.yaml.
    
    Args:
        start_path: Starting path for the search. Defaults to current file's directory.
        
    Returns:
        Absolute path to the project root directory.
    """
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))
    
    current = os.path.abspath(start_path)
    
    # Walk up the directory tree
    for _ in range(10):  # Limit search depth
        paths_yaml = os.path.join(current, "configs", "paths.yaml")
        if os.path.exists(paths_yaml):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    
    # Fallback: assume we're in fastad/ or scripts/ subdirectory
    return os.path.dirname(start_path)


def load_config(config_path: str, resolve: bool = True) -> dict:
    """Load configuration from YAML file with OmegaConf variable interpolation.
    
    This function:
    1. Loads the base paths.yaml configuration
    2. Loads the specified config file
    3. Merges them (config file values override base paths)
    4. Resolves all variable interpolations
    5. Returns a plain dict for backward compatibility
    
    Args:
        config_path: Path to the YAML configuration file.
        resolve: Whether to resolve variable interpolations. Default True.
        
    Returns:
        Configuration as a plain Python dict.
        
    Example:
        config = load_config("configs/MVTec_AD/cable/base.yaml")
        print(config['dataset']['source'])  # /home/11506/datasets/mvtec
    """
    config_path = os.path.abspath(config_path)
    
    # Find project root and paths.yaml
    project_root = find_project_root(os.path.dirname(config_path))
    paths_yaml_path = os.path.join(project_root, "configs", "paths.yaml")
    
    # Load configurations
    if os.path.exists(paths_yaml_path):
        paths_config = OmegaConf.load(paths_yaml_path)
        main_config = OmegaConf.load(config_path)
        # Merge: main_config values take precedence, but can reference paths_config
        config = OmegaConf.merge(paths_config, main_config)
    else:
        # Fallback to simple loading if paths.yaml doesn't exist
        config = OmegaConf.load(config_path)
    
    # Resolve all variable interpolations
    if resolve:
        OmegaConf.resolve(config)
    
    # Convert to plain dict for backward compatibility with existing code
    return OmegaConf.to_container(config, resolve=True)


def load_config_omega(config_path: str, resolve: bool = True) -> DictConfig:
    """Load configuration and return as OmegaConf DictConfig.
    
    Same as load_config but returns OmegaConf object instead of plain dict.
    Useful when you want to use OmegaConf features like dot notation access.
    
    Args:
        config_path: Path to the YAML configuration file.
        resolve: Whether to resolve variable interpolations. Default True.
        
    Returns:
        Configuration as OmegaConf DictConfig.
    """
    config_path = os.path.abspath(config_path)
    
    project_root = find_project_root(os.path.dirname(config_path))
    paths_yaml_path = os.path.join(project_root, "configs", "paths.yaml")
    
    if os.path.exists(paths_yaml_path):
        paths_config = OmegaConf.load(paths_yaml_path)
        main_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(paths_config, main_config)
    else:
        config = OmegaConf.load(config_path)
    
    if resolve:
        OmegaConf.resolve(config)
    
    return config
