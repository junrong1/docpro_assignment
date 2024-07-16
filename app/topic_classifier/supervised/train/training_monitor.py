from transformers import TrainerCallback


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step: {state.global_step}, Loss: {logs['loss']}")