"""
Evaluator for gemmKernel HLS kernel performance
"""
import shutil
import re
from enum import Enum, auto
from pathlib import Path
import time
import openai
from openevolve.utils.code_utils import parse_full_rewrite
# Import hls_eval tools
from hls_eval.tools import (
    CPPCompilerTool,
    VitisHLSCSimTool,
    VitisHLSSynthTool,
    auto_find_vitis_hls_dir
)

# Import EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult
from evolvehls.utils import get_warning_messages, get_error_messages

def evaluate_csim(program_path: str, build_dir: Path = None) -> tuple[bool, EvaluationResult]:
    """
    Evaluate C++ compilation and execution.
    
    Args:
        program_path (str): Path to the program to evaluate
        build_dir (Path): Optional build directory (will create one if not provided)
    
    Returns:
        EvaluationResult: Result of C++ compilation and execution evaluation
    """
    print(f"Evaluating C++ compilation and execution for {program_path}...")
    
    # Get Vitis HLS path
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if not vitis_hls_dir:
        raise EnvironmentError("Vitis HLS not found. Please ensure Vitis HLS is installed and in your PATH.")
    
    # Create C++ compiler tool
    cpp_tool = CPPCompilerTool(vitis_hls_dir)
    
    # Create build directory if not provided
    if build_dir is None:
        build_dir = Path(program_path).parent / "eval_cpp_build"
    build_dir.mkdir(exist_ok=True)
    # add timeout for cpp_too.run
    # try:
    compile_result, run_result = cpp_tool.run(
        build_dir=build_dir,
        source_files=[Path(program_path)],
        build_name="cpp_eval",
        compile_timeout=60,
        execute_timeout=60
    )
    
    comp_return_code = compile_result.data_execution.return_code
    comp_timeout = compile_result.data_execution.timeout
    comp_errors = compile_result.data_execution.stderr

    # check if the program is compiled
    # Check compilation
    if comp_return_code != 0:
        # timeout
        if comp_timeout:
            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": 0.0,
                    "passed_csim": 0.0,
                    "passed_synth": 0.0,
                    "combined_score": 0.25
                },
                artifacts={
                    "info": "Compilation timed out...",
                    "warnings": "Compilation timed out...",
                    "errors": comp_errors
                }
            )
        # compilation error
        else:
            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": 0.0,
                    "passed_csim": 0.0,
                    "passed_synth": 0.0,
                    "combined_score": 0.0
                },
                artifacts={
                    "info": "Compilation failed...",
                    "warnings": "Compilation failed...",
                    "errors": comp_errors
                }
            )
        return False, result
    print("✅ C++ compilation test passed!")
    
    exec_return_code = run_result.data_execution.return_code
    exec_timeout = run_result.data_execution.timeout
    exec_errors = run_result.data_execution.stderr
    exec_output = run_result.data_execution.stdout
    
    # Check execution
    if exec_return_code != 0:
        if exec_timeout:
            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": 1.0,
                    "passed_csim": 0.0,
                    "passed_synth": 0.0,
                    "combined_score": 0.5
                },
                artifacts={
                    "info": "Execution timed out...",
                    "warnings": "Execution timed out...",
                    "errors": exec_errors
                }
            )
        else:
            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": 1.0,
                    "passed_csim": 0.0,
                    "passed_synth": 0.0,
                    "combined_score": 0.5
                },
                artifacts={
                    "info": "Compiled and ran, but test failed",
                    "warnings": "Correctnest test failed...",
                    "errors": exec_output + exec_errors
                }
            )
        return False, result
    
    print("✅ C++ compilation and execution test passed!")
    
    # Return success for C++ evaluation
    return True, EvaluationResult(
        metrics={
            "latency(-)" : 134217736,
            "throughput_interval(-)" : 134217737,
            "speedup": 0.0,
            "passed_compilation": 1.0,
            "passed_csim": 1.0,
            "passed_synth": 0.0,
            "combined_score": 1.0
        },
        artifacts={
            "info": "Passed C++ compilation and execution!",
            "warnings": "",
            "errors": ""
        }
    )

def evaluate_synth(program_path: str, build_dir: Path = None) -> tuple[bool, EvaluationResult]:
    """
    Evaluate HLS synthesis and extract performance metrics.
    
    Args:
        program_path (str): Path to the program to evaluate
        build_dir (Path): Optional build directory (will create one if not provided)
    
    Returns:
        EvaluationResult: Result of synthesis evaluation with performance metrics
    """
    passed_compilation = 1.0
    passed_csim = 1.0
    # Get Vitis HLS path
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if not vitis_hls_dir:
        raise EnvironmentError("Vitis HLS not found. Please ensure Vitis HLS is installed and in your PATH.")
    
    # Create synthesis tool
    synth_tool = VitisHLSSynthTool(vitis_hls_dir)
    
    # Create build directory if not provided
    if build_dir is None:
        build_dir = Path(program_path).parent / "eval_synth_build"
    build_dir.mkdir(exist_ok=True)
    # try:
    result = synth_tool.run(
        build_dir=build_dir,
        source_files=[Path(program_path)],
        build_name="synth_eval",
        hls_top_function="gemmKernel",
        hls_fpga_part="xcu280-fsvh2892-2L-e",
        hls_clock_period_ns=3.33,
        hls_unsafe_math=True,
        timeout=300
    )
    log_file = build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/.autopilot/db/.message_syn.xml"

    
    if result.data_execution.return_code != 0:
        # Check synthesis results
        if result.data_execution.timeout:
            if log_file.exists():
                warnings = get_warning_messages(log_file)
                errors = get_error_messages(log_file)
            else:
                warnings = ""
                errors = "HLS synthesis timed out, and no log file found..."

            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": passed_compilation,
                    "passed_csim": passed_csim,
                    "passed_synth": 0.0,
                    "combined_score": 1.0,
                },
                artifacts={
                    "info": "HLS synthesis timed out...",
                    "warnings": warnings,
                    "errors": errors
                }
            )
        else:
            if log_file.exists():
                warnings = get_warning_messages(log_file)
                errors = get_error_messages(log_file)
            else:
                warnings = ""
                errors = "HLS synthesis failed, and no log file found..."
            result = EvaluationResult(
                metrics={
                    "latency(-)" : 134217736,
                    "throughput_interval(-)" : 134217737,
                    "speedup": 0.0,
                    "passed_compilation": passed_compilation,
                    "passed_csim": passed_csim,
                    "passed_synth": 0.0,
                    "combined_score": 1.0
                },
                artifacts={
                    "info": "Failed to during HLS synthesis",
                    "warnings": warnings,
                    "errors": errors
                }
            )
        return False, result
    
    # Extract synthesis data
    if not result.data_tool:
        raise ValueError("HLS synthesis completed but no data extracted")

    # Extract latency from synthesis data
    # Calculate metrics
    latency = result.data_tool.get('latency_average_cycles', None)
    throughput_interval = result.data_tool.get('throughput_min', None)
    dsp_utilization = result.data_tool.get('resources_dsp_fraction_used', None)
    bram_utilization = result.data_tool.get('resources_bram_fraction_used', None)
    speedup = (134217737 / float(throughput_interval)) + (134217736 / float(latency))
    if latency is None:
        raise ValueError("HLS synthesis completed but no latency found")

    if dsp_utilization > 0.8:
        if log_file.exists():
            warnings = get_warning_messages(log_file)
            errors = get_error_messages(log_file)
        else:
            warnings = "HLS synthesis completed but DSP utilization exceeds 80%, and no log file found..."
            errors = ""
        result = EvaluationResult(
            metrics={
                "latency(-)" : latency,
                "throughput_interval(-)" : throughput_interval,
                "speedup": speedup,
                "passed_compilation": passed_compilation,
                "passed_csim": passed_csim,
                "passed_synth": 0.0,
                "combined_score": 2.0
            },
            artifacts={
                "info": "HLS synthesis completed but DSP utilization exceeds 80%",
                "warnings": warnings,
                "errors": errors
            }
        )
        return False, result
    
    if bram_utilization > 0.8:
        if log_file.exists():
            warnings = get_warning_messages(log_file)
            errors = get_error_messages(log_file)
        else:
            warnings = "HLS synthesis completed but BRAM utilization exceeds 80%, and no log file found..."
            errors = ""
        result = EvaluationResult(
            metrics={
                "latency(-)" : latency,
                "throughput_interval(-)" : throughput_interval,
                "speedup": speedup,
                "passed_compilation": passed_compilation,
                "passed_csim": passed_csim,
                "passed_synth": 0.0,
                "combined_score": 2.0
            },
            artifacts={
                "info": "HLS synthesis completed but BRAM utilization exceeds 80%",
                "warnings": warnings,
                "errors": errors
            }
        )
        return False, result
    
    print(f"✅ HLS synthesis completed! Latency: {latency}")
    warnings = get_warning_messages(log_file)
    return True, EvaluationResult(
        metrics={
            "latency(-)" : latency,
            "throughput_interval(-)" : throughput_interval,
            "speedup": speedup,
            "passed_compilation": passed_compilation,
            "passed_csim": passed_csim,
            "passed_synth": 1.0,
            "combined_score": 3.0 + speedup
        },
        artifacts={
            "info": "HLS synthesis completed successfully!",
            "warnings": warnings,
            "errors": ""
        }
    )

def evaluate_code(program_path: str) -> EvaluationResult:
    """
    Main evaluation function that runs C-simulation and synthesis sequentially.
    
    Args:
        program_path (str): Path to the program to evaluate
    
    Returns:
        EvaluationResult: Combined result of C-simulation and synthesis evaluation
    """
    print(f"Starting full evaluation for {program_path}...")
    
    # Create shared build directory
    file_name = Path(program_path).stem + ".prj"
    build_dir = Path(program_path).parent / file_name
    build_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Evaluate C-simulation
        success, csim_result = evaluate_csim(program_path, build_dir)
        
        # If C-simulation failed, return early
        # if csim_result.metrics["passed_compilation"] == 0.0 or csim_result.metrics["passed_csim"] == 0.0:
        if not success:
            print("❌ C-simulation failed, skipping synthesis")
            return csim_result
        
        # Step 2: Evaluate synthesis
        success, synth_result = evaluate_synth(program_path, build_dir)
        
        # If synthesis failed, return C-simulation result with partial success
        if not success:
            print("❌ Synthesis failed, returning final HLS result")
            return synth_result
        
        # Both passed, return synthesis result (which includes performance metrics)
        print("✅✅✅ Full evaluation completed successfully!")
        return synth_result
        
    except Exception as e:
        return EvaluationResult(
            metrics={
                "latency(-)" : 134217736,
                "throughput_interval(-)" : 134217737,
                "speedup": 0.0,
                "passed_compilation": 0.0,
                "passed_csim": 0.0,
                "passed_synth": 0.0,
                "combined_score": 0.0
            },
            artifacts={
                "info": f"Exception during evaluation: {str(e)}",
                "warnings": "",
                "errors": str(e),
            }
        )
    
    finally:
        log_dir = build_dir / "logs"
        log_paths = {
            "messages.xml" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/.autopilot/db/.message_syn.xml",
            "messages.log" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls.log",
        }
        # make report directory
        report_dir = build_dir / "reports"
        report_paths = {
            "csynth.rpt" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/syn/report/csynth.rpt",
            "csynth.xml" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/syn/report/csynth.xml",
            "gemmKernel_csynth.rpt" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/syn/report/gemmKernel_csynth.rpt",
            "gemmKernel_csynth.xml" : build_dir / "vitis_hls_synth_tool__synth_eval/vitis_hls_synth_tool__synth_eval__proj/solution__synth/syn/report/gemmKernel_csynth.xml",
        }

        # make log_dir and report_dir
        log_dir.mkdir(exist_ok=True)
        report_dir.mkdir(exist_ok=True)
        # copy main csynth reports and logs, and make sure the file exists first
        for log_name, log_path in log_paths.items():
            if log_path.exists():
                shutil.copy(log_path, log_dir / log_name)
        for report_name, report_path in report_paths.items():
            if report_path.exists():
                shutil.copy(report_path, report_dir / report_name)
        # copy the program file to the main build directory
        shutil.copy(program_path, build_dir / "program.cpp")
        try:
            shutil.rmtree(build_dir / "c_compiler_tool__cpp_eval")
            shutil.rmtree(build_dir / "vitis_hls_synth_tool__synth_eval")
        except Exception as e:
            print(f"Warning: Failed to clean up build directory: {e}")

def generate_code(prompt, initial_code):
  try:
      client = openai.OpenAI(
        api_key="dummy",
        base_url="http://172.22.0.0:8850/v1"
      )

      params = {
         "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
         "messages": [
          {"role": "system", "content": 
"""
You are an expert in hardware design and Vitis HLS.
Your task is to generate a C++ code that based on the provided instructions in the prompt.
The output should be purely the C++ code with the testbench, without any other text or comments.

"""          
          },
          {"role": "user", "content": 
f"""
Here is the prompt:
```
{prompt}
```

Here is the initial code:
```cpp
{initial_code}
```
"""
           }
         ],
         "temperature": 0.3
      }
      response = client.chat.completions.create(**params)
      code = parse_full_rewrite(response.choices[0].message.content.strip(), "cpp")
      return code
  except Exception as e:
    raise e

def evaluate(prompt_path: str):
  # step 1: 

  with open("initial_program.cpp") as f:
      initial_code = f.read()

  with open(prompt_path, "r") as f:
    prompt = f.read()

  all_scores = []
  for attempt in range(3):
    code = generate_code(prompt, initial_code)

    # step 2:
    code_path = prompt_path.replace(".md", "_tmp.cpp")
    with open(code_path, "w") as f:
        f.write(code)

    scores = evaluate_code(code_path)
    all_scores.append(scores)
# EvaluationResult(
#                 metrics={
#                     "latency(-)" : LATENCY,
#                     "throughput_interval(-)" : THROUGHPUT,
#                     "speedup": 0.0,
#                     "passed_compilation": 0.0,
#                     "passed_csim": 0.0,
#                     "passed_synth": 0.0,
#                     "combined_score": 0.25
#                 },
#                 artifacts={
#                     "info": "Compilation timed out...",
#                     "warnings": "Compilation timed out...",
#                     "errors": comp_errors
#                 }
#             )
  # average scores
  
  average_latency = sum([score.metrics["latency(-)"] for score in all_scores]) / len(all_scores)
  average_throughput = sum([score.metrics["throughput_interval(-)"] for score in all_scores]) / len(all_scores)
  average_speedup = sum([score.metrics["speedup"] for score in all_scores]) / len(all_scores)
  average_passed_compilation = sum([score.metrics["passed_compilation"] for score in all_scores]) / len(all_scores)
  average_passed_csim = sum([score.metrics["passed_csim"] for score in all_scores]) / len(all_scores)
  average_passed_synth = sum([score.metrics["passed_synth"] for score in all_scores]) / len(all_scores)
  average_combined_score = sum([score.metrics["combined_score"] for score in all_scores]) / len(all_scores)
  highest_combined_score = max([score.metrics["combined_score"] for score in all_scores])

  
  concatenated_warnings = "\n".join([score.artifacts["warnings"] for score in all_scores])
  concatenated_errors = "\n".join([score.artifacts["errors"] for score in all_scores])

  result = EvaluationResult(
      metrics={
        "average_latency(-)" : average_latency,
        "average_throughput_interval(-)" : average_throughput,
        "average_speedup" : average_speedup,
        "average_passed_compilation" : average_passed_compilation,
        "average_passed_csim" : average_passed_csim,
        "average_passed_synth" : average_passed_synth,
        "combined_score" : average_combined_score,
        "highest_combined_score" : highest_combined_score
      },
      artifacts={
          "warnings" : concatenated_warnings,
          "errors" : concatenated_errors
      }
  )
  return result
