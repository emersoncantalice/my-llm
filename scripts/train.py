from my_llm.fine_tuner import FineTuneParams, FineTuner


if __name__ == "__main__":
    trainer = FineTuner()
    output = trainer.run(FineTuneParams())
    print(f"Fine-tuning complete at: {output}")
