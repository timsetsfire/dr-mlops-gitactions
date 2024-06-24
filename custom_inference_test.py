import datarobot as dr 
import yaml 
with open("./model/model-metadata.yaml") as f:
    model_metadata = yaml.load(f, Loader = yaml.SafeLoader)

custom_model_id = model_metadata.get("modelID")
target_type = model_metadata.get("targetType")
for v in dr.enums.TARGET_TYPE.ALL:
    if target_type == v.lower():
        target_type = v
inference_model = model_metadata.get("inferenceModel")
target_name = inference_model.get("targetName")
postive_class_label =  inference_model.get("positiveClassLabel")
negative_class_label = inference_model.get("negativeClassLabel")
environment_id = model_metadata.get("environmentID")

if custom_model_id is None:
    custom_model = dr.CustomInferenceModel.create(
        name = "Arbitrary Classification Model", 
        target_type = target_type, 
        target_name = target_name, 
        language = "python", 
        positive_class_label = postive_class_label, 
        negative_class_label = negative_class_label
    )
else:
    custom_model = dr.CustomInferenceModel.get(custom_model_id)

custom_model_version = dr.CustomModelVersion.create_clean(
    custom_model_id = custom_model.id, base_environment_id = model_metadata["environmentID"], 
    folder_path = "./model", 
)

try: 
    env_build = dr.CustomModelVersionDependencyBuild.start_build(custom_model.id, custom_model_version.id, max_wait = 6000)
except Exception as e:
    print(e)

test_dataset = dr.Dataset.create_from_file(file_path = "test_data.csv")

custom_model_test = dr.CustomModelTest.create(
    custom_model_id = custom_model.id,
    custom_model_version_id = custom_model_version.id,
    dataset_id = test_dataset.id, 
    # max_wait = max_wait
)

print(f"testing overall status: {custom_model_test.overall_status}")
for test, status in custom_model_test.detailed_status.items():
    print(f'{test}: {status["status"]} {status["message"]}')
