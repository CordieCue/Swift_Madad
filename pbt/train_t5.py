from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import numpy as np

def create_trauma_dataset(inputs, labels):
    """Create dataset from trauma descriptions and structured labels."""
    data = [{"text": inp, "label": lbl} for inp, lbl in zip(inputs, labels)]
    return Dataset.from_list(data)

def preprocess_trauma_data(batch):
    """Preprocess trauma data by applying the trauma prompt format."""
    return {
        "input_text": [f"Describe any injuries you have on your head, torso, arms, or legs. Be specific about the location and nature of the injury: {text}" for text in batch["text"]],
        "target_text": [str(label) for label in batch["label"]]
    }

def tokenize_data(examples, tokenizer):
    """Tokenize input and target texts with proper handling of tensors."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=256,
        padding='max_length',
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=64,
            padding='max_length',
            truncation=True
        )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return model_inputs

def train_trauma_model(inputs, labels, output_dir="/media/uas-dtu/mainpcdump/pbt/pbt_weights"):
    """Train T5 model for trauma localization task."""
    try:
        # Create and preprocess dataset
        dataset = create_trauma_dataset(inputs, labels)
        dataset = dataset.map(preprocess_trauma_data, batched=True)

        # Load tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained("/media/uas-dtu/mainpcdump/pbt/t5_span_pretrained/checkpoint-2550")
        model = T5ForConditionalGeneration.from_pretrained("/media/uas-dtu/mainpcdump/pbt/t5_span_pretrained/checkpoint-2550")

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: tokenize_data(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split dataset
        split_data = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

        # Training configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=100,
            weight_decay=0.01,
            logging_dir="./pbt_weights/logs",
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",  # <--- this enables eval during training
            eval_steps=100,
            report_to=["tensorboard"]
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_data["train"],
            eval_dataset=split_data["test"],
            data_collator=data_collator
        )

        # Train and save
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print("Model trained and saved at:", output_dir)
        return model, tokenizer

    except Exception as e:
        print("Error during training:", e)
        raise


# Example usage:
if __name__ == "__main__":
    # Example inputs and labels for training
            # These should be replaced with actual data for a real use case
        input_1 = [
            "My head is fine. There is some bleeding on my chest. My left arm is bent weirdly. I can't see my legs because they’re buried under rubble.",
            "I feel okay all over. No injuries.",
            "There's a missing chunk of my left leg. Arms and torso are fine, head's bleeding a bit."
        ]

        labels_1 = [
            "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
            "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
            "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION"
        ]
        inputs = input_1+[
    "My head’s bleeding a lot, and my vision’s blurry. My chest hurts when I breathe. Arms and legs seem okay, I think.",
    "I’m fine everywhere. No blood, no pain, just shaken up after the crash.",
    "My left arm’s got a deep cut, and it’s bleeding bad. My head’s okay, but my right leg’s pinned under debris.",
    "My leg’s missing after the blast. My head’s spinning, but no blood. Arms and torso are fine.",
    "I can’t feel my legs, they’re buried under rubble. My arms are okay, but my scalp’s cut open.",
    "My chest’s bleeding heavily, I’m pressing on it. My head, arms, and legs are alright, but I’m scared.",
    "My hands are sliced up and bleeding. My right wrist feels broken. Head, torso, and legs are fine.",
    "My forehead’s got blood running down, and I’m dizzy. My torso’s stuck under debris, can’t check it. Arms and legs are okay.",
    "My right thigh’s got a huge gash, blood’s everywhere. My head, arms, and torso are normal.",
    
    "My neck’s bruised and bleeding. My legs and arms are fine. Can’t tell about my back, it’s pinned.",
    "My head’s okay. My stomach’s bleeding and hurts bad when I move. Arms and legs are normal.",
    "My right arm’s gone, just severed in the explosion. My head, torso, and legs are okay.",
    "I don’t feel any injuries. Everything seems fine, no blood or pain.",
    "My left leg’s bleeding profusely, and I can’t move it. My head, torso, and arms are alright.",
    "My arms are cut up and bleeding. My head’s fine, but my legs are stuck under something heavy.",
    "My chest’s got a sharp pain, and there’s blood on my shirt. My head, arms, and legs are okay.",
    "My head’s bleeding from a cut on my temple. My torso, arms, and legs are normal, I think.",
    "My left hand’s crushed and bleeding. My head, torso, and legs are fine after the crash.",
    "I can’t see my torso, it’s covered by debris. My head, arms, and legs feel okay.",
    
    "My right leg’s got multiple cuts, bleeding a lot. My head, torso, and arms are normal.",
    "My head’s okay. My left arm’s twisted bad, can’t move it. Torso and legs are fine.",
    "My back’s bleeding, I can feel it. My arms, legs, and head are alright, but I’m freaking out.",
    "My legs are fine. My right hand’s bleeding and feels broken. Head and torso are okay.",
    "My head’s got a gash, and I’m seeing stars. My arms, torso, and legs are normal.",
    "My left arm’s missing after the blast. My head, torso, and legs seem okay.",
    "I can’t move my right leg, it’s bleeding bad. My head, torso, and arms are fine.",
    "My face is cut and bleeding. Everything else feels normal, thank God.",
    "My torso’s pinned, can’t check it. My head’s bleeding slightly. Arms and legs are okay.",
    "My arms are okay, but I can’t feel my legs at all. Head and torso seem fine.",
    
    "My head’s normal. My chest’s bleeding, and it hurts to breathe. Arms and legs are okay.",
    "My right arm’s fractured, and it’s bleeding. My head, torso, and legs are fine.",
    "My left leg’s stuck under debris, can’t move it. My head, torso, and arms are normal.",
    "I’m bleeding from my scalp, and my left arm’s cut bad. My torso and legs are okay.",
    "My head’s fine. My stomach’s got a deep cut, bleeding a lot. Arms and legs are normal.",
    "My right foot’s bleeding and feels broken. My head, torso, and arms are okay.",
    "My arms are trapped under rubble, can’t check them. My head, torso, and legs are fine.",
    "My head’s bleeding, and I’m dizzy. My torso’s okay, but my left leg’s bleeding bad.",
    "My chest’s fine. My right arm’s missing. My head and legs are normal after the explosion.",
    "My legs are bleeding from cuts. My head’s okay, but my torso’s pinned under debris.",
    
    "My head’s normal. My left hand’s bleeding and crushed. My torso and legs are fine.",
    "My right thigh’s torn open, blood’s pouring out. My head, arms, and torso are okay.",
    "I can’t feel my arms, they’re stuck. My head’s bleeding slightly. Torso and legs are okay.",
    "My back’s hurting bad, and there’s blood. My head, arms, and legs are normal.",
    "My head’s fine. My left leg’s bleeding and won’t move. My torso and arms are okay.",
    "My right arm’s bleeding and feels broken. My head, torso, and legs are normal.",
    "My scalp’s cut open, bleeding bad. My torso, arms, and legs are fine, I think.",
    "My chest’s bleeding, and it’s hard to breathe. My head, arms, and legs are okay.",
    "My left arm’s gone after the crash. My head, torso, and legs are normal.",
    "I can’t check my legs, they’re buried. My head’s bleeding, and my arms are fine.",
    
    "My head’s okay. My stomach’s bleeding bad. My arms and legs are normal.",
    "My right hand’s bleeding and hurts a lot. My head, torso, and legs are fine.",
    "My legs are stuck under debris, can’t feel them. My head, torso, and arms are okay.",
    "My head’s bleeding from a cut. My torso’s fine, but my right arm’s bleeding bad.",
    "My chest’s got a gash, bleeding heavily. My head, arms, and legs are normal.",
    "My left leg’s missing after the blast. My head, torso, and arms are okay.",
    "My arms are fine. My head’s bleeding, and my torso’s pinned under rubble.",
    "My right leg’s bleeding and feels broken. My head, torso, and arms are normal.",
    "My head’s fine. My left arm’s cut and bleeding. My torso and legs are okay.",
    "My back’s bleeding, I can feel it. My head, arms, and legs are normal.",
    
    "My head’s bleeding, and I’m woozy. My torso’s okay, but my legs are trapped.",
    "My right arm’s bleeding and won’t move. My head, torso, and legs are fine.",
    "My chest’s fine. My left leg’s bleeding bad. My head and arms are normal.",
    "My head’s normal. My right hand’s crushed and bleeding. My torso and legs are okay.",
    "My left thigh’s got a deep cut, bleeding a lot. My head, torso, and arms are fine.",
    "My arms are bleeding and stuck under debris. My head and legs are okay.",
    "My head’s bleeding from my forehead. My torso’s fine, but my right leg’s bleeding.",
    "My chest’s bleeding bad, hurts to breathe. My head, arms, and legs are okay.",
    "My right arm’s missing. My head, torso, and legs are normal after the crash.",
    "My legs are fine. My head’s bleeding slightly. My torso’s pinned under debris.",
    
    "My head’s okay. My left arm’s bleeding and feels broken. My torso and legs are fine.",
    "My right thigh’s bleeding heavily. My head, torso, and arms are normal.",
    "My head’s fine. My chest’s bleeding and hurts bad. My arms and legs are okay.",
    "My left hand’s bleeding and crushed. My head, torso, and legs are normal.",
    "My legs are trapped, can’t check them. My head’s bleeding, and my arms are fine.",
    "My head’s normal. My stomach’s bleeding bad. My arms and legs are okay.",
    "My right arm’s bleeding and fractured. My head, torso, and legs are fine.",
    "My head’s bleeding from a cut. My torso’s fine, but my left leg’s bleeding bad.",
    "My chest’s got a deep cut, bleeding a lot. My head, arms, and legs are normal.",
    "My left leg’s missing after the explosion. My head, torso, and arms are okay.",
    
    "My arms are okay. My head’s bleeding, and my torso’s stuck under debris.",
    "My right leg’s bleeding and feels broken. My head, torso, and arms are normal.",
    "My head’s fine. My left arm’s cut and bleeding bad. My torso and legs are okay.",
    "My back’s bleeding, I can feel it. My head, arms, and legs are normal.",
    "My head’s bleeding, and I’m dizzy. My torso’s okay, but my legs are pinned.",
    "My right arm’s bleeding and won’t move. My head, torso, and legs are fine.",
    "My chest’s fine. My left leg’s bleeding bad. My head and arms are normal.",
    "My head’s normal. My right hand’s crushed and bleeding. My torso and legs are okay.",
    "My left thigh’s got a deep cut, bleeding a lot. My head, torso, and arms are fine.",
    "My arms are bleeding and trapped under debris. My head and legs are okay.",
    
    "My head’s bleeding from my forehead. My torso’s fine, but my right leg’s bleeding.",
    "My chest’s bleeding bad, hurts to breathe. My head, arms, and legs are okay.",
    "My right arm’s missing. My head, torso, and legs are normal after the crash.",
    "My legs are fine. My head’s bleeding slightly. My torso’s pinned under debris.",
    "My head’s okay. My left arm’s bleeding and feels broken. My torso and legs are fine.",
    "My right thigh’s bleeding heavily. My head, torso, and arms are normal.",
    "My head’s fine. My chest’s bleeding and hurts bad. My arms and legs are okay.",
    "My left hand’s bleeding and crushed. My head, torso, and legs are normal.",
    "My right foot is crushed and bleeding. Head, torso, and arms are okay.",
    "I can't feel my arms, they are pinned under a heavy beam. My head is bleeding a little. Legs and torso seem fine.",
    
    "There's a large burn on my chest. Everything else seems untouched.",
    "My left leg is gone below the knee. My right arm has a deep gash. Head and torso are fine.",
    "I hit my head hard, it's throbbing. My back hurts badly. Arms and legs are okay.",
    "All my limbs are trapped. I can't see any injuries on them. My head and torso are clear.",
    "A piece of shrapnel hit my abdomen. My head feels fuzzy. Arms and legs are fine.",
    "My vision is blurry and my scalp is cut. My left hand is broken. Torso and legs are okay.",
    "I think my ribs are broken, it's hard to breathe. My legs are scraped up. Head and arms are fine.",
    "No visible injuries on my head, torso, arms or legs. Just feeling shaken.",
    "My right arm is completely severed. My left leg is also badly mangled and bleeding.",
    "My neck has a deep cut and is bleeding. My chest also has a cut. Arms and legs are fine.",
    
    "My head has a small scratch, no bleeding. My torso, arms and legs are fine.",
    "Both my legs are trapped under the car. I can't see them. My head is bleeding from a gash. Arms and torso are okay.",
    "My left hand is missing a few fingers. My right ankle is twisted and swollen. Head and torso are clear.",
    "I have a puncture wound in my side. My head is fine. My arms are scraped. My legs are okay.",
    "My left arm is broken and my right leg is also broken. Head and torso are surprisingly okay.",
    "There's a burn mark on my forehead. My left shin is bleeding. Torso and arms are fine.",
    "I landed on my back, it hurts a lot. My head hit the ground, I see stars. Arms and legs seem fine.",
    "My right leg is amputated at the hip. My left arm has minor cuts. Head and torso are fine.",
    "I can't assess my torso or legs, they are under debris. My head is fine. My arms have some bruises.",
    "My face is cut up badly. My chest feels tight but no visible wound. Arms and legs are okay.",
    "Both my hands are severely burned. My legs are fine. My head and torso are also fine.",
    "My left foot is gone. My right arm is trapped and I can't see it. Head and torso are okay.",
    
    "My ears are oozing blood but my body feels fine.",
    "There's fresh blood trickling from both ear canals, otherwise I'm okay.",
    "My abdomen is in intense pain, yet there are no cuts or bruises.",
    "I have severe stomach cramps but no external bleeding or lacerations.",
    "My thighs are covered in dark bruises, nothing else is injured.",
    "There are painful contusions on both legs, the rest of me feels normal.",
    "My palms are numb—I can’t sense anything in my hands.",
    "I can’t detect any feeling in my fingers or wrists.",
    "Both my legs have been severed, and the bleeding has stopped.",
    "I no longer have my legs; there’s no more blood flow.",
    
    "My right foot is bent at an odd angle and bleeding profusely.",
    "My right ankle is dislocated and blood is pouring out.",
    "Something struck my skull, and now my memory is blank.",
    "I took a blow to the head and can’t remember anything since.",
    "My abdomen is puffy and throbbing with pain.",
    "My belly feels distended and it hurts terribly.",
    "My right hand is missing after the explosion.",
    "They say my right hand was completely severed off.",  
    "I have a deep slice across my brow that's gushing blood.",
    "My forehead is lacerated and blood keeps pouring out.",

    "Debris is pinning my legs and I have no sensation below my hips.",
    "My lower limbs are trapped under concrete and they feel numb.",
    "Skin is the only thing holding my hand; it’s nearly detached.",
    "My hand is dangling by a flap of skin after it was trapped.",
    "I got struck hard in the skull and sternum, but my limbs feel uninjured.",
    "Both my head and chest took the impact; my arms and legs seem okay.",
    "There’s blood gushing from both of my legs and I am unable to move them.",
    "My legs are pouring blood and completely immobile.",
    "A jagged metal shard is embedded in my abdomen.",
    "I feel a sharp object lodged deep in my chest.",
    
    "My lower body is completely numb; I feel nothing below my hips.",
    "There’s zero sensation in my legs or feet.",
    "Both my arms and legs are lacerated, but my chest is uninjured.",
    "I have slashes on my legs and arms; there’s no damage to my torso.",
    "My palms are stiff and unresponsive, yet there’s no bleeding.",
    "I can’t move my hands—they’re locked up, though they aren’t bleeding.",
    "I have a ripping pain in my abdomen and spine.",
    "My belly and lower back feel like they've been shredded.",
    "There are deep bruises on my arms and chest, no blood visible.",
    "My chest and arms are black and blue, yet there’s no bleeding.",

    "My left thigh is smashed and I have injuries on both arms.",
    "The left side of my leg is flattened, and my arms are wounded too.",
    "I have a ripping pain in my abdomen and spine.",
    "My belly and lower back feel like they've been shredded.",
    "There are deep bruises on my arms and chest, no blood visible.",
    "My chest and arms are black and blue, yet there’s no bleeding.",
    "My left thigh is smashed and I have injuries on both arms.",
    "The left side of my leg is flattened, and my arms are wounded too.",
    "There’s blood pouring from my eye and it’s scorching with pain.",
    "My eye won’t stay open because it’s bleeding so badly.",
    
    "My lower back aches so fiercely I can’t straighten up.",
    "I have excruciating pain in my back that prevents me from sitting.",
    "My chest is soaked in blood and I’m lightheaded.",
    "There’s bleeding across my chest and I feel faint and woozy.",
    "I have no fingers on my left hand—they were ripped away.",
    "My scalp is cut and my hair is matted with blood.",
    "I’ve got a head injury and blood is soaked into my hair.",
    "My right hand was severed, and I have no other visible injuries.",
    "I’m missing my right hand; everything else seems fine.",
    
        ]


        labels = labels_1 + [    
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NOT TESTABLE, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: AMPUTATION",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL", # Assuming neck injury maps to Head
    "Head: WOUND, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: AMPUTATION", # Chest tight but no visible wound -> NORMAL
    "Head: NORMAL, Torso: NOT TESTABLE, Upper Extremities: WOUND, Lower Extremities: NOT TESTABLE",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: AMPUTATION",
    
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",  
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: AMPUTATION",


    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",

    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NOT TESTABLE",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: WOUND",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: NOT TESTABLE, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",


    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: WOUND, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: WOUND, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: WOUND, Torso: NORMAL, Upper Extremities: NORMAL, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL",
    "Head: NORMAL, Torso: NORMAL, Upper Extremities: AMPUTATION, Lower Extremities: NORMAL"
]
        


        train_trauma_model(inputs, labels)
