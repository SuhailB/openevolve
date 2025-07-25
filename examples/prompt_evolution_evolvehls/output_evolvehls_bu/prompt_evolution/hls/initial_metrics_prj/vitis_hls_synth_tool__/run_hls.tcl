open_project vitis_hls_synth_tool____proj
add_files initial_program.cpp
open_solution solution__synth -flow_target vivado
set_top gemmKernel
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.33 -name clk_default
config_compile -unsafe_math_optimizations
csynth_design
exit
