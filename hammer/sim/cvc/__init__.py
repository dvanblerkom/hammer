#  hammer-vlsi plugin for cvc
#
#  See LICENSE for license details.

from hammer.vlsi import HammerSimTool, HammerToolStep, HammerLSFSubmitCommand, HammerLSFSettings
from hammer.common.cvc import CvcTool
from hammer.logging import HammerVLSILogging

from typing import Dict, List, Optional, Callable, Tuple

from hammer.vlsi import FlowLevel, TimeValue

import hammer.utils as hammer_utils
import hammer.tech as hammer_tech
from hammer.tech import HammerTechnologyUtils

import os
import re
import shutil
import json
from multiprocessing import Process

class CVC(HammerSimTool, CvcTool):

    def tool_config_prefix(self) -> str:
        return "sim.cvc"

    def fill_outputs(self) -> bool:
        # TODO: support automatic waveform generation in a similar fashion to SAIFs
        self.output_waveforms = []
        self.output_saifs = []
        self.output_top_module = self.top_module
        self.output_tb_name = self.get_setting("sim.inputs.tb_name")
        self.output_tb_dut = self.get_setting("sim.inputs.tb_dut")
        self.output_level = self.get_setting("sim.inputs.level")
        if self.get_setting("sim.inputs.saif.mode") != "none":
            if not self.benchmarks:
                self.output_saifs.append(os.path.join(self.run_dir, "ucli.saif"))
            for benchmark in self.benchmarks:
                self.output_saifs.append(os.path.join(self.benchmark_run_dir(benchmark), "ucli.saif"))
        return True

    @property
    def steps(self) -> List[HammerToolStep]:
        return self.make_steps_from_methods([
            self.write_gl_files,
            self.fix_verilog_sdf,
            self.run_cvc,
            self.run_simulation
            ])

    def benchmark_run_dir(self, bmark_path: str) -> str:
        """Generate a benchmark run directory."""
        # TODO(ucb-bar/hammer#462) this method should be passed the name of the bmark rather than its path
        bmark = os.path.basename(bmark_path)
        return os.path.join(self.run_dir, bmark)

    @property
    def force_regs_file_path(self) -> str:
        return os.path.join(self.run_dir, "force_regs.ucli")

    @property
    def access_tab_file_path(self) -> str:
        return os.path.join(self.run_dir, "access.tab")

    @property
    def simulator_executable_path(self) -> str:
        return os.path.join(self.run_dir, "simv")

    @property
    def run_tcl_path(self) -> str:
        return os.path.join(self.run_dir, "run.tcl")

    @property
    def env_vars(self) -> Dict[str, str]:
        v = dict(super().env_vars)
        return v

    def get_verilog_models(self) -> List[str]:
        verilog_sim_files = self.technology.read_libs([
            hammer_tech.filters.verilog_sim_filter
        ], hammer_tech.HammerTechnologyUtils.to_plain_item)
        return verilog_sim_files

    def fix_verilog_sdf(self) -> bool:
        if self.level.is_gatelevel():
            abspath_input_files = list(map(lambda name: os.path.join(os.getcwd(), name), self.input_files))
            for vfilename in abspath_input_files:
                with open(vfilename, 'r') as sf:
                    vfilename_fix = vfilename+"_fix.v"
                    with open(vfilename_fix, 'w') as df:
                        for line in sf:
                            line = line.replace('SUM','SUM_');
                            df.write(line)

            if self.sdf_file:
                sfilename = os.path.join(os.getcwd(), self.sdf_file)
                with open(sfilename, 'r') as sf:
                    sfilename_fix = sfilename+"_fix.sdf"
                    with open(sfilename_fix, 'w') as df:
                        for line in sf:
                            line = line.replace('SUM','SUM_');
                            df.write(line)

        return True

    def write_gl_files(self) -> bool:
        if self.level == FlowLevel.RTL:
            return True

        if (1):
            return True

        tb_prefix = self.get_setting("sim.inputs.tb_dut")
        force_val = self.get_setting("sim.inputs.gl_register_force_value")

        abspath_seq_cells = os.path.join(os.getcwd(), self.seq_cells)
        if not os.path.isfile(abspath_seq_cells):
            self.logger.error("List of seq cells json not found as expected at {0}".format(self.seq_cells))

        with open(self.access_tab_file_path, "w") as f:
            with open(abspath_seq_cells) as seq_file:
                seq_json = json.load(seq_file)
                assert isinstance(seq_json, List), "list of all sequential cells should be a json list of strings not {}".format(type(seq_json))
                for cell in seq_json:
                    f.write("acc=wn:{cell_name}\n".format(cell_name=cell))

        abspath_all_regs = os.path.join(os.getcwd(), self.all_regs)
        if not os.path.isfile(abspath_all_regs):
            self.logger.error("List of all regs json not found as expected at {0}".format(self.all_regs))

        with open(self.force_regs_file_path, "w") as f:
            with open(abspath_all_regs) as reg_file:
                reg_json = json.load(reg_file)
                assert isinstance(reg_json, List), "list of all sequential cells should be a json list of dictionaries from string to string not {}".format(type(reg_json))
                for reg in sorted(reg_json, key=lambda r: len(r["path"])): # TODO: This is a workaround for a bug in P-2019.06
                    path = reg["path"]
                    path = '.'.join(path.split('/'))
                    pin = reg["pin"]
                    f.write("force -deposit {" + tb_prefix + "." + path + " ." + pin + "} " + str(force_val) + "\n")

        return True

    def run_cvc(self) -> bool:
        # run through inputs and append to CL arguments
        cvc_bin = self.get_setting("sim.cvc.cvc_bin")
        if not os.path.isfile(cvc_bin):
          self.logger.error("CVC binary not found as expected at {0}".format(cvc_bin))
          return False

        if not self.check_input_files([".v", ".v.gz", ".sv", ".so", ".cc", ".c"]):
          return False

        # We are switching working directories and we still need to find paths
        abspath_input_files = list(map(lambda name: os.path.join(os.getcwd(), name), self.input_files))
        if self.level.is_gatelevel():
            abspath_input_files = [fix+"_fix.v" for fix in abspath_input_files]

        top_module = self.top_module
        # TODO(johnwright) sanity check the timescale string
        timescale = self.get_setting("sim.inputs.timescale")
        options = self.get_setting("sim.inputs.options", [])
        defines = self.get_setting("sim.inputs.defines", [])
        access_tab_filename = self.access_tab_file_path
        tb_name = self.get_setting("sim.inputs.tb_name")

        # Build args
        args = [
          cvc_bin,
          "+interp",
          "+dumpvars"
        ]

#        if timescale is not None:
#            args.append('-timescale={}'.format(timescale))

        # black box options
        args.extend(options)

        # Add in all input files
        args.extend(abspath_input_files)

        # Note: we always want to get the verilog models because most real designs will instantate a few
        # tech-specific cells in the source RTL (IO cells, clock gaters, etc.)

#        args.extend(self.get_verilog_models())

        pat = self.get_setting("sim.inputs.cvcvlib")
        pat2 = self.get_setting("sim.inputs.cvclib")
        for model in self.get_verilog_models():
            if (re.search(pat,model)):
                args.extend(['-v', model])
            elif (re.search(pat2,model)):
                args.extend([model])
#            else:
#                args.extend([model])

        for define in defines:
            args.extend(['+define+' + define])

        if self.level.is_gatelevel():
#            args.extend(['-P'])
#            args.extend([access_tab_filename])
            if self.get_setting("sim.inputs.timing_annotated"):
#                args.extend(["+neg_tchk"])
                args.extend(["+sdfverbose"])
#                args.extend(["-negdelay"])
                args.extend(["+maxdelays"])
                args.extend(["+sdf_annotate"])
                if self.sdf_file:
                    args.extend(["{sdf}+{tb}.{top}".format(tb=tb_name, top=top_module, sdf=os.path.join(os.getcwd(), self.sdf_file)+"_fix.sdf")])
            else:
                args.extend(["+notimingcheck"])
#                args.extend(["+delay_mode_zero"])
        else:
            # Also disable timing at RTL level for any hard macros
            args.extend(["+notimingcheck"])
#            args.extend(["+delay_mode_zero"])


#        if tb_name != "":
#            args.extend(["-top", tb_name])

        args.extend(['-o', self.simulator_executable_path])

        HammerVLSILogging.enable_colour = False
        HammerVLSILogging.enable_tag = False

        # Delete an old copy of the simulator if it exists
        if os.path.exists(self.simulator_executable_path):
            os.remove(self.simulator_executable_path)

        # Remove the csrc directory (otherwise the simulator will be stale)
        if os.path.exists(os.path.join(self.run_dir, "csrc")):
            shutil.rmtree(os.path.join(self.run_dir, "csrc"))

        # Generate a simulator
        self.run_executable(args, cwd=self.run_dir)

        HammerVLSILogging.enable_colour = True
        HammerVLSILogging.enable_tag = True

#        return os.path.exists(self.simulator_executable_path)
        return True

    def run_simulation(self) -> bool:
        if not self.get_setting("sim.inputs.execute_sim"):
            self.logger.warning("Not running any simulations because sim.inputs.execute_sim is unset.")
            return True

        top_module = self.top_module
        exec_flags_prepend = self.get_setting("sim.inputs.execution_flags_prepend", [])
        exec_flags = self.get_setting("sim.inputs.execution_flags", [])
        exec_flags_append = self.get_setting("sim.inputs.execution_flags_append", [])
        force_regs_filename = self.force_regs_file_path
        tb_prefix = self.get_setting("sim.inputs.tb_dut")
        saif_mode = self.get_setting("sim.inputs.saif.mode")
        saif_start_time: Optional[str] = None
        saif_end_time: Optional[str] = None
        saif_start_trigger_raw: Optional[str] = None
        saif_end_trigger_raw: Optional[str] = None
        if saif_mode == "time":
            saif_start_time = self.get_setting("sim.inputs.saif.start_time")
            saif_end_time = self.get_setting("sim.inputs.saif.end_time")
        elif saif_mode == "trigger":
            self.logger.error("Trigger SAIF mode currently unsupported.")
        elif saif_mode == "trigger_raw":
            saif_start_trigger_raw = self.get_setting("sim.inputs.saif.start_trigger_raw")
            saif_end_trigger_raw = self.get_setting("sim.inputs.saif.end_trigger_raw")
        elif saif_mode == "full":
            pass
        elif saif_mode == "none":
            pass
        else:
            self.logger.warning("Bad saif_mode:${saif_mode}. Valid modes are time, trigger, full, or none. Defaulting to none.")
            saif_mode = "none"

        if self.level == FlowLevel.RTL and saif_mode != "none":
            find_regs_run_tcl = []
            if saif_mode != "none":
                stime: Optional[TimeValue] = None
                if saif_mode == "time":
                    assert saif_start_time
                    stime = TimeValue(saif_start_time)
                    find_regs_run_tcl.append("run {start}ns".format(start=stime.value_in_units("ns")))
                elif saif_mode == "trigger_raw":
                    find_regs_run_tcl.append(saif_start_trigger_raw)
                    find_regs_run_tcl.append("run")
                elif saif_mode == "full":
                    pass
                # start saif
                find_regs_run_tcl.append("power {dut}".format(dut=tb_prefix))
                find_regs_run_tcl.append("config endofsim noexit")
                if saif_mode == "time":
                    assert saif_end_time
                    assert stime
                    etime = TimeValue(saif_end_time)
                    find_regs_run_tcl.append("run {end}ns".format(end=(etime.value_in_units("ns") - stime.value_in_units("ns"))))
                elif saif_mode == "trigger_raw":
                    find_regs_run_tcl.append(saif_end_trigger_raw)
                    find_regs_run_tcl.append("run")
                elif saif_mode == "full":
                    find_regs_run_tcl.append("run")
                # stop saif
                find_regs_run_tcl.append("power -report ucli.saif 1e-9 {dut}".format(dut=tb_prefix))
            find_regs_run_tcl.append("run")
            find_regs_run_tcl.append("exit")
            self.write_contents_to_path("\n".join(find_regs_run_tcl), self.run_tcl_path)

        if self.level.is_gatelevel():
            find_regs_run_tcl = []
            find_regs_run_tcl.append("source " + force_regs_filename)
            if saif_mode != "none":
                stime: Optional[TimeValue] = None
                if saif_mode == "time":
                    assert saif_start_time
                    stime = TimeValue(saif_start_time)
                    find_regs_run_tcl.append("run {start}ns".format(start=stime.value_in_units("ns")))
                elif saif_mode == "trigger_raw":
                    find_regs_run_tcl.append(saif_start_trigger_raw)
                    find_regs_run_tcl.append("run")
                elif saif_mode == "full":
                    pass
                # start saif
                find_regs_run_tcl.append("power -gate_level on")
                find_regs_run_tcl.append("power {dut}".format(dut=tb_prefix))
                find_regs_run_tcl.append("config endofsim noexit")
                if saif_mode == "time":
                    assert saif_end_time
                    assert stime
                    etime = TimeValue(saif_end_time)
                    find_regs_run_tcl.append("run {end}ns".format(end=(etime.value_in_units("ns") - stime.value_in_units("ns"))))
                elif saif_mode == "trigger_raw":
                    find_regs_run_tcl.append(saif_end_trigger_raw)
                    find_regs_run_tcl.append("run")
                elif saif_mode == "full":
                    find_regs_run_tcl.append("run")
                # stop saif
                find_regs_run_tcl.append("power -report ucli.saif 1e-9 {dut}".format(dut=tb_prefix))
            find_regs_run_tcl.append("run")
            find_regs_run_tcl.append("exit")
            self.write_contents_to_path("\n".join(find_regs_run_tcl), self.run_tcl_path)

        cvc_bin = self.get_setting("sim.cvc.cvc_bin")
        for benchmark in self.benchmarks:
            if not os.path.isfile(benchmark):
              self.logger.error("benchmark not found as expected at {0}".format(benchmark))
              return False

        # setup simulation arguments
        args = [ self.simulator_executable_path ]
        args.extend(exec_flags_prepend)
        if self.get_setting("sim.cvc.fgp") and self.version() >= self.version_number("M-2017.03"):
            # num_threads is in addition to a master thread, so reduce by 1
            num_threads=int(self.get_setting("vlsi.core.max_threads")) - 1
            args.append("-fgp=num_threads:{threads},num_fsdb_threads:0,allow_less_cores,dynamictoggle".format(threads=max(num_threads,1)))
        args.extend(exec_flags)
        if self.level.is_gatelevel():
            if saif_mode != "none":
                args.extend([
                    # Reduce the number ucli instructions by auto starting and auto stopping
                    '-saif_opt+toggle_start_at_set_region+toggle_stop_at_toggle_report',
                    # Only needed if we are using start time pruning so we can return to ucli after endofsim
                    '-ucli2Proc',
                ])
            args.extend(["-ucli", "-do", self.run_tcl_path])
        elif self.level == FlowLevel.RTL and saif_mode != "none":
            args.extend([
                # Reduce the number ucli instructions by auto starting and auto stopping
                '-saif_opt+toggle_start_at_set_region+toggle_stop_at_toggle_report',
                # Only needed if we are using start time pruning so we can return to ucli after endofsim
                '-ucli2Proc',
            ])
            args.extend(["-ucli", "-do", self.run_tcl_path])
        args.extend(exec_flags_append)

        HammerVLSILogging.enable_colour = False
        HammerVLSILogging.enable_tag = False

        # Our current invocation of CVC is only using a single core
        if isinstance(self.submit_command, HammerLSFSubmitCommand):
            old_settings = self.submit_command.settings._asdict()
            del old_settings['num_cpus']
            self.submit_command.settings = HammerLSFSettings(num_cpus=1, **old_settings)

        # Run the simulations in as many parallel runs as the user wants
        if self.get_setting("sim.inputs.parallel_runs") == 0:
            runs = 1
        else:
            runs = self.get_setting("sim.inputs.parallel_runs")
        bp = [] #  type: List[Process]
        running = 0
        ran = 0
        for benchmark in self.benchmarks:
            bmark_run_dir = self.benchmark_run_dir(benchmark)
            # Make the rundir if it does not exist
            hammer_utils.mkdir_p(bmark_run_dir)
            if runs > 0 and running >= runs: # We are currently running the maximum number so we join first
                bp[ran].join()
                ran = ran + 1
                running = running - 1
#            bp.append(Process(target=self.run_executable, args=(args + [benchmark],), kwargs={'cwd':bmark_run_dir}))
#            bp[-1].start()
            running = running + 1
        # Make sure we join all remaining runs
        for p in bp:
            p.join()


#        if self.benchmarks == []:
#            self.run_executable(args, cwd=self.run_dir)

        HammerVLSILogging.enable_colour = True
        HammerVLSILogging.enable_tag = True

        return True

tool = CVC
