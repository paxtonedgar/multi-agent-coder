"""
Test CLI entry points and user interactions
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main, _show_routing_stats, _list_available_models, _update_model_discovery, _configure_hf_routing

class TestCLIEntryPoints:
    """Test CLI entry points and argument parsing"""
    
    def test_basic_task_execution(self, temp_brain, cli_inputs):
        """Test basic task execution via CLI"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {
                'final_result': 'Task completed successfully',
                'status': 'success'
            }
            
            # Mock sys.argv
            with patch('sys.argv', ['main.py'] + cli_inputs['basic_task']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            # Verify workflow was called
            mock_workflow.assert_called_once()
            call_args = mock_workflow.call_args
            assert call_args[1]['task'] == 'Create a simple calculator'
            assert call_args[1]['mode'] == 'full'
    
    def test_task_with_options(self, temp_brain, cli_inputs):
        """Test task execution with various options"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            with patch('sys.argv', ['main.py'] + cli_inputs['task_with_options']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            call_args = mock_workflow.call_args
            assert call_args[1]['task'] == 'Build a web app'
            assert call_args[1]['mode'] == 'quick'
    
    def test_task_with_repository(self, temp_brain, cli_inputs):
        """Test task execution with repository URL"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            with patch('sys.argv', ['main.py'] + cli_inputs['task_with_repo']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            call_args = mock_workflow.call_args
            assert call_args[1]['repo_url'] == 'https://github.com/test/repo'
    
    def test_help_request(self, cli_inputs):
        """Test help command"""
        with patch('sys.argv', ['main.py'] + cli_inputs['help_request']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
    
    def test_stats_request(self, temp_brain, cli_inputs):
        """Test stats command"""
        with patch('main._show_routing_stats') as mock_stats:
            with patch('sys.argv', ['main.py'] + cli_inputs['stats_request']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            mock_stats.assert_called_once_with(temp_brain)
    
    def test_models_request(self, temp_brain, cli_inputs):
        """Test list models command"""
        with patch('main._list_available_models') as mock_models:
            with patch('sys.argv', ['main.py'] + cli_inputs['models_request']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            mock_models.assert_called_once_with(temp_brain)
    
    def test_update_models_flag(self, temp_brain):
        """Test update models flag"""
        with patch('main._update_model_discovery') as mock_update:
            with patch('main.run_workflow') as mock_workflow:
                mock_workflow.return_value = {'final_result': 'Success'}
                
                with patch('sys.argv', ['main.py', 'Test task', '--update-models']):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
                
                mock_update.assert_called_once_with(temp_brain)
    
    def test_hf_routing_configuration(self, temp_brain):
        """Test HF routing configuration"""
        with patch('main._configure_hf_routing') as mock_config:
            with patch('main.run_workflow') as mock_workflow:
                mock_workflow.return_value = {'final_result': 'Success'}
                
                with patch('sys.argv', ['main.py', 'Test task', '--use-hf', 'auto-reasoning']):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
                
                mock_config.assert_called_once_with(temp_brain, 'auto-reasoning', False)
    
    def test_force_hf_flag(self, temp_brain):
        """Test force HF flag"""
        with patch('main._configure_hf_routing') as mock_config:
            with patch('main.run_workflow') as mock_workflow:
                mock_workflow.return_value = {'final_result': 'Success'}
                
                with patch('sys.argv', ['main.py', 'Test task', '--use-hf', 'DeepSeek-V3', '--force-hf']):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
                
                mock_config.assert_called_once_with(temp_brain, 'DeepSeek-V3', True)

class TestCLIErrorHandling:
    """Test CLI error handling and edge cases"""
    
    def test_missing_task_argument(self):
        """Test error when no task is provided"""
        with patch('sys.argv', ['main.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
    
    def test_invalid_mode(self):
        """Test error with invalid mode"""
        with patch('sys.argv', ['main.py', 'Test task', '--mode', 'invalid']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
    
    def test_workflow_failure(self, temp_brain):
        """Test handling of workflow failures"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.side_effect = Exception("Workflow failed")
            
            with patch('sys.argv', ['main.py', 'Test task']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1
    
    def test_api_key_validation(self, temp_brain):
        """Test API key validation"""
        # Test with no API keys
        with patch.dict(os.environ, {}, clear=True):
            with patch('main.run_workflow') as mock_workflow:
                mock_workflow.return_value = {'final_result': 'Success'}
                
                with patch('sys.argv', ['main.py', 'Test task']):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        # Should not raise exception, should use mock model
                        main()
                
                mock_workflow.assert_called_once()

class TestCLISubprocess:
    """Test CLI via subprocess for end-to-end testing"""
    
    def test_cli_subprocess_basic(self):
        """Test CLI execution via subprocess"""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert 'usage:' in result.stdout.lower()
        assert 'Multi-Agent Coding System' in result.stdout
    
    def test_cli_subprocess_invalid_args(self):
        """Test CLI with invalid arguments"""
        result = subprocess.run(
            [sys.executable, 'main.py', '--invalid-flag'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode != 0
        assert 'error:' in result.stderr.lower()
    
    @pytest.mark.asyncio
    async def test_cli_subprocess_with_mock(self, temp_brain):
        """Test CLI execution with mocked workflow"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Mock success'}
            
            # Create a temporary script to test
            temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            temp_script.write("""
import sys
sys.path.insert(0, '.')
from main import main
if __name__ == '__main__':
    main()
""")
            temp_script.close()
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_script.name, 'Test task'],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent,
                    env={**os.environ, 'OPENAI_API_KEY': 'sk-test-placeholder-for-testing'}
                )
                
                # Should succeed with mock
                assert result.returncode == 0
                assert 'Workflow completed successfully' in result.stdout
                
            finally:
                os.unlink(temp_script.name)

class TestCLIUserScenarios:
    """Test realistic user interaction scenarios"""
    
    @pytest.mark.parametrize("scenario", [
        "simple_code_generation",
        "complex_project", 
        "bug_fix"
    ])
    def test_user_scenario_workflow(self, temp_brain, user_scenarios, scenario):
        """Test complete user scenario workflows"""
        scenario_data = user_scenarios[scenario]
        
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {
                'final_result': f'{scenario} completed successfully',
                'status': 'success'
            }
            
            with patch('sys.argv', ['main.py', scenario_data['initial_prompt']]):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            # Verify workflow was called with correct task
            call_args = mock_workflow.call_args
            assert call_args[1]['task'] == scenario_data['initial_prompt']
    
    def test_multi_turn_interaction(self, temp_brain):
        """Test multi-turn user interaction simulation"""
        tasks = [
            "Create a simple calculator",
            "Add error handling to the calculator",
            "Add unit tests for the calculator",
            "Create documentation for the calculator"
        ]
        
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            for i, task in enumerate(tasks):
                with patch('sys.argv', ['main.py', task]):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
                
                # Verify each call
                assert mock_workflow.call_count == i + 1
                call_args = mock_workflow.call_args
                assert call_args[1]['task'] == task
    
    def test_iterative_refinement(self, temp_brain):
        """Test iterative refinement scenario"""
        base_task = "Create a REST API"
        refinements = [
            "Add authentication",
            "Add rate limiting", 
            "Add caching",
            "Add monitoring"
        ]
        
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            # Initial task
            with patch('sys.argv', ['main.py', base_task]):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            # Refinements
            for refinement in refinements:
                refined_task = f"{base_task} with {refinement}"
                with patch('sys.argv', ['main.py', refined_task]):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
            
            # Verify all calls were made
            assert mock_workflow.call_count == len(refinements) + 1

class TestCLIPerformance:
    """Test CLI performance and response times"""
    
    def test_cli_startup_time(self, temp_brain):
        """Test CLI startup performance"""
        import time
        
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            start_time = time.time()
            with patch('sys.argv', ['main.py', 'Test task']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            end_time = time.time()
            
            # Should start up quickly
            assert end_time - start_time < 2.0
    
    def test_cli_memory_usage(self, temp_brain):
        """Test CLI memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            with patch('sys.argv', ['main.py', 'Test task']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024

class TestCLIIntegration:
    """Test CLI integration with other components"""
    
    def test_cli_with_hf_routing(self, temp_brain):
        """Test CLI with HF routing enabled"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            with patch('sys.argv', ['main.py', 'Test task', '--use-hf', 'auto-reasoning']):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            # Verify HF routing was configured
            assert 'hf_routing_config' in temp_brain.memory
    
    def test_cli_with_git_integration(self, temp_brain, test_repo):
        """Test CLI with Git repository integration"""
        with patch('main.run_workflow') as mock_workflow:
            mock_workflow.return_value = {'final_result': 'Success'}
            
            repo_url = f"file://{test_repo}"
            with patch('sys.argv', ['main.py', 'Analyze repository', '--repo-url', repo_url]):
                with patch('main.ProjectBrain', return_value=temp_brain):
                    main()
            
            call_args = mock_workflow.call_args
            assert call_args[1]['repo_url'] == repo_url
    
    def test_cli_mode_selection(self, temp_brain):
        """Test different CLI modes"""
        modes = ['full', 'quick', 'research']
        
        for mode in modes:
            with patch('main.run_workflow') as mock_workflow:
                mock_workflow.return_value = {'final_result': 'Success'}
                
                with patch('sys.argv', ['main.py', 'Test task', '--mode', mode]):
                    with patch('main.ProjectBrain', return_value=temp_brain):
                        main()
                
                call_args = mock_workflow.call_args
                assert call_args[1]['mode'] == mode 