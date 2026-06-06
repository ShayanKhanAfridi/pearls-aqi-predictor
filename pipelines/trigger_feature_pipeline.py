# ============================================================
# Trigger Feature Pipeline Job on Hopsworks
# ============================================================

import os
import sys
import hopsworks

def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY", "")
    if not api_key:
        print("❌ HOPSWORKS_API_KEY environment variable is missing!")
        sys.exit(1)

    print("🔗 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=api_key
        )
        print("✅ Logged in successfully.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    job_name = "feature_pipeline_job"
    print(f"🚀 Retrieving job '{job_name}'...")
    try:
        job_api = project.get_job_api()
        job = job_api.get_job(job_name)
        if job is None:
            raise ValueError(f"Job '{job_name}' not found in the project.")
    except Exception as e:
        print(f"❌ Failed to retrieve job '{job_name}': {e}")
        print("\n💡 HOW TO FIX THIS ERROR:")
        print("1. Go to your Hopsworks project Console UI: https://eu-west.cloud.hopsworks.ai")
        print("2. Click on the 'Jobs' tab in the left sidebar.")
        print("3. Click 'New Job', set Type to 'Python', and name it exactly: feature_pipeline_job")
        print("4. Upload pipelines/feature_pipeline.py as the file.")
        print("5. Save the Job. Once created, this GitHub Action will trigger it successfully!\n")
        sys.exit(1)

    print(f"▶ Triggering job '{job_name}'...")
    try:
        execution = job.run()
        print(f"✅ Job run triggered successfully. Execution ID: {execution.id}")
        print(f"📊 Monitor the job run here: https://eu-west.cloud.hopsworks.ai/p/{project.id}/jobs")
    except Exception as e:
        print(f"❌ Failed to trigger job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
