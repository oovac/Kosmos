#!/usr/bin/env python3
"""
Validate Northflank deployment template.

This script validates the northflank.json template file
to ensure it's correctly formatted before deployment.
"""

import json
import sys
from pathlib import Path


def validate_northflank_template(template_path: str) -> tuple[bool, list[str]]:
    """
    Validate a Northflank template file.
    
    Args:
        template_path: Path to the northflank.json file
        
    Returns:
        Tuple of (is_valid, list of errors/warnings)
    """
    errors = []
    warnings = []
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {template_path}"]
    
    # Check required top-level fields
    required_fields = ["apiVersion", "name", "spec"]
    for field in required_fields:
        if field not in template:
            errors.append(f"Missing required field: {field}")
    
    # Validate apiVersion
    if template.get("apiVersion") != "v1":
        warnings.append(f"Unexpected apiVersion: {template.get('apiVersion')} (expected 'v1')")
    
    # Validate spec
    spec = template.get("spec", {})
    if spec.get("kind") != "Workflow":
        warnings.append(f"Unexpected spec.kind: {spec.get('kind')} (expected 'Workflow')")
    
    # Validate steps
    steps = spec.get("spec", {}).get("steps", [])
    if not steps:
        errors.append("No steps defined in workflow")
    else:
        step_kinds = [step.get("kind") for step in steps]
        
        # Check for required resource types
        if "SecretGroup" not in step_kinds:
            warnings.append("No SecretGroup defined - secrets may not be configured")
        
        if "BuildService" not in step_kinds:
            errors.append("No BuildService defined - cannot build Docker image")
        
        if "DeploymentService" not in step_kinds:
            errors.append("No DeploymentService defined - nothing to deploy")
        
        # Validate each step
        for i, step in enumerate(steps):
            step_errors = validate_step(step, i)
            errors.extend(step_errors)
    
    # Validate arguments
    arguments = template.get("arguments", {})
    if "ANTHROPIC_API_KEY" not in arguments:
        warnings.append("ANTHROPIC_API_KEY not defined in arguments")
    
    is_valid = len(errors) == 0
    messages = [f"ERROR: {e}" for e in errors] + [f"WARNING: {w}" for w in warnings]
    
    return is_valid, messages


def validate_step(step: dict, index: int) -> list[str]:
    """Validate a workflow step."""
    errors = []
    kind = step.get("kind", "Unknown")
    spec = step.get("spec", {})
    
    if "kind" not in step:
        errors.append(f"Step {index}: Missing 'kind' field")
        return errors
    
    if "spec" not in step:
        errors.append(f"Step {index} ({kind}): Missing 'spec' field")
        return errors
    
    # Validate based on kind
    if kind == "DeploymentService":
        if "name" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'name'")
        if "deployment" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'deployment'")
        if "ports" not in spec or not spec["ports"]:
            errors.append(f"Step {index} ({kind}): No ports defined")
        if "healthChecks" not in spec:
            errors.append(f"Step {index} ({kind}): No health checks defined (recommended)")
    
    elif kind == "BuildService":
        if "name" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'name'")
        if "vcsData" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'vcsData' (repository info)")
        if "buildSettings" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'buildSettings'")
    
    elif kind == "Addon":
        if "name" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'name'")
        if "type" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'type'")
        addon_type = spec.get("type", "")
        if addon_type not in ["postgresql", "redis", "mongodb", "mysql", "minio"]:
            errors.append(f"Step {index} ({kind}): Unknown addon type: {addon_type}")
    
    elif kind == "SecretGroup":
        if "name" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'name'")
        if "secrets" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'secrets'")
    
    elif kind == "Job":
        if "name" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'name'")
        if "deployment" not in spec:
            errors.append(f"Step {index} ({kind}): Missing 'deployment'")
    
    return errors


def main():
    """Main entry point."""
    # Find northflank.json
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    template_path = repo_root / "northflank.json"
    
    if len(sys.argv) > 1:
        template_path = Path(sys.argv[1])
    
    print(f"Validating Northflank template: {template_path}")
    print("=" * 60)
    
    is_valid, messages = validate_northflank_template(str(template_path))
    
    if messages:
        for msg in messages:
            print(msg)
        print()
    
    if is_valid:
        print("✓ Template is valid!")
        print()
        print("Next steps:")
        print("1. Push changes to GitHub")
        print("2. Deploy to Northflank:")
        print("   - Use one-click deploy button, or")
        print("   - Run: northflank template create --file northflank.json")
        return 0
    else:
        print("✗ Template validation failed!")
        print("Please fix the errors above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

