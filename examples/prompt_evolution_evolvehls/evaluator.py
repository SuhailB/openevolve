from openevolve.evaluation_result import EvaluationResult
from evolvehls.config import EvolveHLSConfig
from evolvehls import EvolveHLS
import os
import json
import asyncio
import subprocess

class PromptEvaluator:
  def __init__(self, config_path: str):
    self.config = EvolveHLSConfig.from_yaml(config_path)


  def run(self, program_path: str):
    # get file name without extension
    program_name_no_extension = os.path.splitext(os.path.basename(program_path))[0]
    self.config.output_dir = os.path.join(self.config.output_dir, program_name_no_extension)
    self.best_program_info_path = os.path.join(self.config.output_dir, "best", "best_program_info.json")
    self.config.evaluator_path = os.path.join(self.config.output_dir, "evaluator.py")
    with open(program_path, 'r') as f:
      self.config.system_message = f.read()

    # Save the updated config to a temporary YAML file for subprocess
    import tempfile
    import yaml
    config_dict = self.config.to_dict()
    print(config_dict)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
      yaml.dump(self.config.to_dict(), tmp_config)
      tmp_config_path = tmp_config.name
    
    with open(tmp_config_path, "r") as f:
      print(f.read())

    try:
      print(f"Starting EvolveHLS for {self.config.iterations} iterations (subprocess)")
      subprocess.run([
        "evolvehls-opt", "-c", tmp_config_path
      ], check=True)
      with open(self.best_program_info_path, "r") as f:
        best_program_info = json.load(f)
      with open(os.path.join(self.config.output_dir, "metrics.json"), "r") as f:
        iteration_metrics = json.load(f)

      combined_score = sum([metric["combined_score"] for metric in iteration_metrics]) / len(iteration_metrics)
      speedup = sum([metric["speedup"] for metric in iteration_metrics]) / len(iteration_metrics)
      passed_compilation = sum([metric["passed_compilation"] for metric in iteration_metrics]) / len(iteration_metrics)
      passed_csim = sum([metric["passed_csim"] for metric in iteration_metrics]) / len(iteration_metrics)
      passed_synth = sum([metric["passed_synth"] for metric in iteration_metrics]) / len(iteration_metrics)
      highest_score = best_program_info["metrics"]["combined_score"]
      result = EvaluationResult(
        metrics={
          "highest_score" : highest_score,
          "average_speedup" : speedup,
          "average_passed_compilation" : passed_compilation,
          "average_passed_csim" : passed_csim,
          "average_passed_synth" : passed_synth,
          "combined_score" : combined_score
        },
        artifacts={
          "info" : "Prompt Evolution Completed",
        }
      )
      return result
    except Exception as e:
      result = EvaluationResult(
        metrics={
          "highest_score" : 0.0,
          "average_speedup" : 0.0,
          "average_passed_compilation" : 0.0,
          "average_passed_csim" : 0.0,
          "average_passed_synth" : 0.0,
          "combined_score" : 0.0
        },
        artifacts={
          "info" : f"Prompt Evolution Failed: {str(e)}",
        }
      )
      return result


def evaluate(program_path: str) -> EvaluationResult:
  prompt_evaluator = PromptEvaluator("/home/suhailb/Projects/EvolveHLS/extern/openevolve/examples/prompt_evolution_evolvehls/evolvehls_config.yaml")
  return prompt_evaluator.run(program_path)