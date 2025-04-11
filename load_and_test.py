import load_CIFAR10 as ld
import trainer
import models as mdl

DATA_DIR = r"E:\cifar-10-batches-py"    # 数据集路径
MODEL_DIR = r"saved_model_full_5680"


if __name__ == "__main__":
    _, _, test_X, test_Y = ld.load_cifar10(DATA_DIR)
    trained_mlp = mdl.SimpleModel.load(MODEL_DIR)

    testing_loss, testing_accuracy = trainer.run_evaluation(test_X, test_Y, trained_mlp)
    print(f"testing_loss: {testing_loss}, testing_accuracy: {testing_accuracy}")
