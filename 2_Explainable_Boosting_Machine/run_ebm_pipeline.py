import argparse
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "0_Shared_Data_And_Splits"))
import ebm_workflow as workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-profile-scoring", action="store_true")
    parser.add_argument("--shared", type=Path, default=workflow.SHARED)
    parser.add_argument("--cefr-output", type=Path, default=BASE / "2_CEFR_Classification")
    parser.add_argument("--level-output", type=Path, default=BASE / "3_Level_Prediction")
    parser.add_argument("--llm-output", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    options, extra = parser.parse_known_args()
    if any(arg == "--output" or arg.startswith("--output=") for arg in extra):
        parser.error("Use --cefr-output and --level-output for the two separate tasks.")
    extra += ["--shared", str(options.shared), "--cefr-output", str(options.cefr_output),
              "--level-output", str(options.level_output)]
    llm = (options.llm_output or options.shared.resolve().parent / "5_Data_For_LLM_Post_Training").resolve()
    if options.baseline_output:extra += ["--baseline-output",str(options.baseline_output)]
    extra += ["--llm-output",str(llm)]
    workflow.organize_saved_outputs()
    workflow.main("cefr", ["prepare", *extra])
    shared = options.shared.resolve()
    manifest, _ = workflow.verify_shared(shared)
    if not options.skip_profile_scoring and (manifest["scope"] != "full_corpus" or workflow.resolve_manifest_path(manifest["data_directory"]) != workflow.DATA):
        raise ValueError("Custom/small datasets require --skip-profile-scoring; profile scoring uses the complete thesis dataframe.")
    sealed = (shared / "final_evaluation.json").exists()
    if not sealed:
        workflow.main("cefr", ["rank", *extra])
        for task in ("cefr", "level"):
            workflow.main(task, ["train", "--use-ranked", *extra])
    for task in ("cefr", "level"):
        workflow.main(task, ["report", *extra])
    workflow.main("cefr", ["summarize", *extra])
    if not options.skip_profile_scoring and not sealed:
        profile = BASE / "5_Data_For_LLM_Post_Training/2_Level_Reference_Profiles"
        sys.path.insert(0, str(profile))
        import build_level_profiles
        for task, name in (
            ("level", "2_Level_Reference_Profiles"),
            ("cefr", "1_CEFR_Reference_Profiles"),
        ):
            build_level_profiles.main([
                "--profile-target", task,
                "--split-file", str(shared / "splits.csv"),
                "--ebm-dir", str(llm / name / "3_Reusable_Models_For_LLMs"),
                "--output-dir", str(llm / name),
            ])


if __name__ == "__main__":
    main()
