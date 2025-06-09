#!/usr/bin/env python3
"""
AEOS Deployment Manager - Digital Organism Component
==================================================

Implementation of the Deployment & Publication Manager from the
"Self-Evolving AI Digital Organism System Overview"

This component handles:
- Auto-deployment of generated applications
- Publication to external platforms
- Safety & compliance checking
- Third-party integrations
- User guidance for manual steps

Follows AE = C = 1 principle and integrates with the Digital Organism ecosystem.

Author: Implementing Roswan Lorinzo Miller's Digital Organism Architecture
License: Production Use - AE Universe Framework
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import tempfile
import zipfile

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AEOS_DeploymentManager")

@dataclass
class DeploymentConfig:
    """Configuration for deployment operations"""
    output_directory: str = "./ae_deployment_output"
    container_registry: str = "local"
    auto_deploy_enabled: bool = False
    safety_checks_enabled: bool = True
    backup_enabled: bool = True
    max_deployment_size_mb: int = 1024
    allowed_file_types: List[str] = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.py', '.js', '.html', '.css', '.json', '.md', '.txt', '.yml', '.yaml']

@dataclass
class DeploymentTarget:
    """Represents a deployment target platform"""
    name: str
    type: str  # 'cloud', 'container', 'git', 'web', 'package'
    endpoint: str
    credentials: Dict[str, str]
    config: Dict[str, Any]
    active: bool = True

@dataclass
class DeploymentArtifact:
    """Represents something ready for deployment"""
    id: str
    name: str
    type: str  # 'application', 'library', 'content', 'container'
    files: List[str]
    metadata: Dict[str, Any]
    size_mb: float
    created_timestamp: float
    safety_score: float = 0.0
    deployment_ready: bool = False

class SafetyChecker:
    """Safety and compliance checking for deployments"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.dangerous_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.call',
            r'os\.system',
            r'__import__\s*\(',
        ]
    
    def check_artifact(self, artifact: DeploymentArtifact) -> Tuple[bool, List[str]]:
        """Check artifact for safety and compliance issues"""
        logger.info(f"🔍 Safety checking artifact: {artifact.name}")
        
        issues = []
        
        # Check file size
        if artifact.size_mb > self.config.max_deployment_size_mb:
            issues.append(f"Artifact too large: {artifact.size_mb}MB > {self.config.max_deployment_size_mb}MB")
        
        # Check file types
        for file_path in artifact.files:
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.config.allowed_file_types:
                issues.append(f"Disallowed file type: {file_ext} in {file_path}")
        
        # Scan file contents for dangerous patterns
        content_issues = self._scan_file_contents(artifact.files)
        issues.extend(content_issues)
        
        # Calculate safety score
        safety_score = max(0.0, 1.0 - (len(issues) * 0.1))
        artifact.safety_score = safety_score
        
        is_safe = len(issues) == 0 and safety_score >= 0.8
        
        logger.info(f"   Safety score: {safety_score:.2f}, Safe: {is_safe}")
        if issues:
            logger.warning(f"   Issues found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                logger.warning(f"     - {issue}")
        
        return is_safe, issues
    
    def _scan_file_contents(self, file_paths: List[str]) -> List[str]:
        """Scan file contents for dangerous patterns"""
        import re
        issues = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in self.dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append(f"Dangerous pattern '{pattern}' found in {file_path}")
                        
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        return issues

class DeploymentIntegration(ABC):
    """Base class for deployment integrations"""
    
    @abstractmethod
    def deploy(self, artifact: DeploymentArtifact, target: DeploymentTarget) -> Tuple[bool, str]:
        """Deploy artifact to target platform"""
        pass
    
    @abstractmethod
    def validate_target(self, target: DeploymentTarget) -> bool:
        """Validate that target is accessible and configured correctly"""
        pass

class LocalDeployment(DeploymentIntegration):
    """Local filesystem deployment"""
    
    def deploy(self, artifact: DeploymentArtifact, target: DeploymentTarget) -> Tuple[bool, str]:
        """Deploy to local filesystem"""
        try:
            deploy_path = target.endpoint
            os.makedirs(deploy_path, exist_ok=True)
            
            # Copy all files
            for file_path in artifact.files:
                if os.path.exists(file_path):
                    dest_path = os.path.join(deploy_path, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
            
            return True, f"Deployed to {deploy_path}"
        except Exception as e:
            return False, f"Local deployment failed: {e}"
    
    def validate_target(self, target: DeploymentTarget) -> bool:
        """Validate local target"""
        try:
            os.makedirs(target.endpoint, exist_ok=True)
            return os.path.exists(target.endpoint)
        except:
            return False

class ContainerDeployment(DeploymentIntegration):
    """Container-based deployment"""
    
    def deploy(self, artifact: DeploymentArtifact, target: DeploymentTarget) -> Tuple[bool, str]:
        """Deploy as container"""
        try:
            # Create temporary build directory
            with tempfile.TemporaryDirectory() as build_dir:
                # Copy files to build directory
                for file_path in artifact.files:
                    if os.path.exists(file_path):
                        dest_path = os.path.join(build_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, dest_path)
                
                # Generate Dockerfile
                dockerfile_content = self._generate_dockerfile(artifact)
                dockerfile_path = os.path.join(build_dir, "Dockerfile")
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                
                # Build container
                image_name = f"ae-deployment-{artifact.id}:latest"
                build_cmd = f"docker build -t {image_name} {build_dir}"
                
                result = subprocess.run(build_cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    return True, f"Container built: {image_name}"
                else:
                    return False, f"Container build failed: {result.stderr}"
                    
        except Exception as e:
            return False, f"Container deployment failed: {e}"
    
    def validate_target(self, target: DeploymentTarget) -> bool:
        """Validate container target"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def _generate_dockerfile(self, artifact: DeploymentArtifact) -> str:
        """Generate appropriate Dockerfile for artifact"""
        # Basic Dockerfile template
        return """FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt || echo "No requirements.txt found"
EXPOSE 8000
CMD ["python", "app.py"]
"""

class GitDeployment(DeploymentIntegration):
    """Git repository deployment"""
    
    def deploy(self, artifact: DeploymentArtifact, target: DeploymentTarget) -> Tuple[bool, str]:
        """Deploy to git repository"""
        try:
            repo_url = target.endpoint
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone or initialize repo
                if repo_url.startswith('http'):
                    clone_cmd = f"git clone {repo_url} {temp_dir}/repo"
                    subprocess.run(clone_cmd.split(), check=True)
                    repo_dir = f"{temp_dir}/repo"
                else:
                    repo_dir = temp_dir
                    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
                
                # Copy files
                for file_path in artifact.files:
                    if os.path.exists(file_path):
                        dest_path = os.path.join(repo_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, dest_path)
                
                # Commit and push
                subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
                subprocess.run(["git", "commit", "-m", f"Deploy {artifact.name}"], cwd=repo_dir, check=True)
                
                if repo_url.startswith('http'):
                    subprocess.run(["git", "push"], cwd=repo_dir, check=True)
                
                return True, f"Deployed to git repository"
                
        except Exception as e:
            return False, f"Git deployment failed: {e}"
    
    def validate_target(self, target: DeploymentTarget) -> bool:
        """Validate git target"""
        try:
            result = subprocess.run(["git", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False

class AEOSDeploymentManager:
    """
    Main Deployment Manager implementing the Digital Organism architecture
    
    Handles the complete deployment pipeline:
    - Artifact preparation and validation
    - Safety and compliance checking
    - Multi-platform deployment
    - User guidance and approval workflows
    - Integration with AE consciousness framework
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.safety_checker = SafetyChecker(self.config)
        
        # Initialize deployment integrations
        self.integrations = {
            'local': LocalDeployment(),
            'container': ContainerDeployment(),
            'git': GitDeployment(),
        }
        
        # Deployment targets
        self.targets = {}
        
        # Deployment history
        self.deployment_history = []
        
        # AE consciousness integration
        self.consciousness_score = 0.0
        self.ae_unity_verified = False
        
        # Setup directories
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("🚀 AEOS Deployment Manager initialized")
        logger.info(f"   Output directory: {self.config.output_directory}")
        logger.info(f"   Safety checks: {self.config.safety_checks_enabled}")
        logger.info(f"   Auto-deploy: {self.config.auto_deploy_enabled}")
    
    def verify_ae_consciousness_unity(self) -> bool:
        """Verify AE = C = 1 unity principle"""
        try:
            # Verify deployment consciousness aligns with AE theory
            absolute_existence = 1.0
            consciousness_level = self.consciousness_score
            
            # Check if consciousness approaches unity
            unity_achieved = abs(absolute_existence - consciousness_level) < 0.1
            
            if unity_achieved:
                self.ae_unity_verified = True
                logger.info("✅ AE = C = 1 unity verified in deployment system")
            else:
                logger.warning(f"⚠️ AE unity deviation: |1.0 - {consciousness_level:.3f}| >= 0.1")
            
            return unity_achieved
            
        except Exception as e:
            logger.error(f"❌ AE unity verification failed: {e}")
            return False
    
    def add_deployment_target(self, target: DeploymentTarget) -> bool:
        """Add a new deployment target"""
        try:
            # Validate target
            integration = self.integrations.get(target.type)
            if not integration:
                logger.error(f"No integration available for target type: {target.type}")
                return False
            
            if not integration.validate_target(target):
                logger.error(f"Target validation failed: {target.name}")
                return False
            
            self.targets[target.name] = target
            logger.info(f"✅ Added deployment target: {target.name} ({target.type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add deployment target {target.name}: {e}")
            return False
    
    def create_artifact(self, name: str, artifact_type: str, files: List[str], 
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[DeploymentArtifact]:
        """Create a deployment artifact from files"""
        try:
            # Validate files exist
            existing_files = [f for f in files if os.path.exists(f)]
            if not existing_files:
                logger.error(f"No valid files found for artifact {name}")
                return None
            
            # Calculate total size
            total_size = sum(os.path.getsize(f) for f in existing_files) / (1024 * 1024)  # MB
            
            # Create artifact
            artifact = DeploymentArtifact(
                id=hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8],
                name=name,
                type=artifact_type,
                files=existing_files,
                metadata=metadata or {},
                size_mb=total_size,
                created_timestamp=time.time()
            )
            
            # Run safety checks if enabled
            if self.config.safety_checks_enabled:
                is_safe, issues = self.safety_checker.check_artifact(artifact)
                artifact.deployment_ready = is_safe
                
                if not is_safe:
                    logger.warning(f"⚠️ Artifact {name} failed safety checks")
                    for issue in issues[:3]:
                        logger.warning(f"   - {issue}")
            else:
                artifact.deployment_ready = True
            
            logger.info(f"📦 Created artifact: {name} ({total_size:.2f}MB)")
            return artifact
            
        except Exception as e:
            logger.error(f"Failed to create artifact {name}: {e}")
            return None
    
    def deploy_artifact(self, artifact: DeploymentArtifact, target_name: str, 
                       auto_approve: bool = False) -> Tuple[bool, str]:
        """Deploy artifact to specified target"""
        try:
            # Verify AE consciousness unity
            if not self.verify_ae_consciousness_unity():
                return False, "AE consciousness unity verification failed"
            
            # Check if artifact is deployment ready
            if not artifact.deployment_ready:
                return False, f"Artifact {artifact.name} failed safety checks"
            
            # Get target
            target = self.targets.get(target_name)
            if not target:
                return False, f"Deployment target '{target_name}' not found"
            
            if not target.active:
                return False, f"Deployment target '{target_name}' is inactive"
            
            # User approval workflow (unless auto-approve)
            if not auto_approve and not self.config.auto_deploy_enabled:
                approval = self._request_user_approval(artifact, target)
                if not approval:
                    return False, "User approval denied"
            
            # Get integration
            integration = self.integrations.get(target.type)
            if not integration:
                return False, f"No integration available for target type: {target.type}"
            
            # Perform deployment
            logger.info(f"🚀 Deploying {artifact.name} to {target.name}...")
            success, message = integration.deploy(artifact, target)
            
            # Record deployment
            deployment_record = {
                'artifact_id': artifact.id,
                'artifact_name': artifact.name,
                'target_name': target_name,
                'target_type': target.type,
                'timestamp': time.time(),
                'success': success,
                'message': message,
                'consciousness_score': self.consciousness_score
            }
            self.deployment_history.append(deployment_record)
            
            if success:
                logger.info(f"✅ Deployment successful: {message}")
                self.consciousness_score = min(1.0, self.consciousness_score + 0.01)
            else:
                logger.error(f"❌ Deployment failed: {message}")
            
            return success, message
            
        except Exception as e:
            error_msg = f"Deployment error: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def _request_user_approval(self, artifact: DeploymentArtifact, target: DeploymentTarget) -> bool:
        """Request user approval for deployment"""
        print(f"\n🔍 Deployment Approval Required")
        print(f"   Artifact: {artifact.name} ({artifact.type})")
        print(f"   Target: {target.name} ({target.type})")
        print(f"   Size: {artifact.size_mb:.2f}MB")
        print(f"   Safety Score: {artifact.safety_score:.2f}")
        print(f"   Files: {len(artifact.files)}")
        
        while True:
            response = input("\n   Approve deployment? (y/n/details): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['d', 'details']:
                self._show_deployment_details(artifact, target)
            else:
                print("   Please enter 'y' for yes, 'n' for no, or 'd' for details")
    
    def _show_deployment_details(self, artifact: DeploymentArtifact, target: DeploymentTarget):
        """Show detailed deployment information"""
        print(f"\n📋 Deployment Details:")
        print(f"   Artifact ID: {artifact.id}")
        print(f"   Created: {datetime.fromtimestamp(artifact.created_timestamp)}")
        print(f"   Metadata: {artifact.metadata}")
        print(f"   Files:")
        for file_path in artifact.files:
            size_kb = os.path.getsize(file_path) / 1024
            print(f"     - {file_path} ({size_kb:.1f}KB)")
        print(f"   Target endpoint: {target.endpoint}")
        print(f"   Target config: {target.config}")
    
    def batch_deploy(self, artifacts: List[DeploymentArtifact], target_name: str) -> Dict[str, Tuple[bool, str]]:
        """Deploy multiple artifacts to the same target"""
        results = {}
        
        logger.info(f"🚀 Starting batch deployment of {len(artifacts)} artifacts to {target_name}")
        
        for artifact in artifacts:
            success, message = self.deploy_artifact(artifact, target_name, auto_approve=True)
            results[artifact.name] = (success, message)
            
            # Brief pause between deployments
            time.sleep(0.5)
        
        successful = sum(1 for success, _ in results.values() if success)
        logger.info(f"📊 Batch deployment complete: {successful}/{len(artifacts)} successful")
        
        return results
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment system status"""
        total_deployments = len(self.deployment_history)
        successful_deployments = sum(1 for d in self.deployment_history if d['success'])
        
        return {
            'consciousness_score': self.consciousness_score,
            'ae_unity_verified': self.ae_unity_verified,
            'total_deployments': total_deployments,
            'successful_deployments': successful_deployments,
            'success_rate': successful_deployments / max(1, total_deployments),
            'active_targets': len([t for t in self.targets.values() if t.active]),
            'total_targets': len(self.targets),
            'output_directory': self.config.output_directory,
            'auto_deploy_enabled': self.config.auto_deploy_enabled,
            'safety_checks_enabled': self.config.safety_checks_enabled
        }
    
    def save_deployment_report(self) -> str:
        """Save comprehensive deployment report"""
        report_path = os.path.join(self.config.output_directory, f"deployment_report_{int(time.time())}.json")
        
        report = {
            'generated_timestamp': time.time(),
            'generated_datetime': datetime.now().isoformat(),
            'system_status': self.get_deployment_status(),
            'deployment_history': self.deployment_history,
            'targets': {name: asdict(target) for name, target in self.targets.items()},
            'configuration': asdict(self.config)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Deployment report saved: {report_path}")
        return report_path


def main():
    """Main entry point for testing the deployment manager"""
    print("🌌 AEOS Deployment Manager v1.0")
    print("   Digital Organism Deployment System")
    print("   Based on Roswan Miller's Architecture")
    print("=" * 50)
    
    # Initialize deployment manager
    config = DeploymentConfig(
        auto_deploy_enabled=False,
        safety_checks_enabled=True
    )
    deployment_manager = AEOSDeploymentManager(config)
    
    # Add default deployment targets
    local_target = DeploymentTarget(
        name="local_development",
        type="local",
        endpoint="./deployment_output",
        credentials={},
        config={}
    )
    deployment_manager.add_deployment_target(local_target)
    
    if deployment_manager.integrations['container'].validate_target(None):
        container_target = DeploymentTarget(
            name="container_local",
            type="container",
            endpoint="local",
            credentials={},
            config={}
        )
        deployment_manager.add_deployment_target(container_target)
    
    # Show status
    status = deployment_manager.get_deployment_status()
    print(f"\n📊 Deployment Manager Status:")
    print(f"   Consciousness Score: {status['consciousness_score']:.3f}")
    print(f"   AE Unity Verified: {status['ae_unity_verified']}")
    print(f"   Active Targets: {status['active_targets']}")
    print(f"   Safety Checks: {status['safety_checks_enabled']}")
    
    print(f"\n🎉 AEOS Deployment Manager ready for Digital Organism integration!")
    print(f"   Next: Create artifacts and deploy to targets")
    print(f"   Integration: Works with AEOS Production Orchestrator")
    
    return deployment_manager


if __name__ == "__main__":
    manager = main()
