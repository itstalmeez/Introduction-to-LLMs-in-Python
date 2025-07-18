    from transformers import pipline

    # Language Generation
    generator = pipeline(task='text-generation', model='distilgpt2')
    prompt = 'The Gion neighborhod in kyoto ius famous for'
    output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id)
    print(output[0]["generated_text"])



    # Language Translation
    translator = pipeline(task='translation_en_to_es', model='modelname')
    text='blah sooooo'
    output = translator(text, clean_up_tokenization_spaces=True)
    print(output[0]['translation_text'])




    # Encoder_Only
    llm=pipline(model='bert')
    print(llm.model)
    print(llm.model.config)
    (llm.model.config.is_decoder)
    (llm.model.config.is_encoder_decoder)




    # Decoder-Only
    llm=pipline(model='gpt')
    print(llm.model.config)
    (llm.model.config.is_decoder)




    # EncoderDecoder
    llm = pipeline(model='helsinki...')
    print(llm.model)
    print(llm.model.config)
    (llm.model.config.is_encoder_decoder)




    # Loading a dataset for fine-tuning
    from datasets import load_dataset
    train_data = load_dataset('imdb', split='train')
    train_data = data.shard(num_shards=4, index=0)
    test_data = load_dataset('imdb', split='test')
    test_data = data.shard(num_shards=4, index=0)




    # Auto Classes
    from transformers import AutoModel, AutoTokenizer
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



    # Tokenization
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset

    train_data = load_dataset('imdb', split='train')
    train_data = data.shard(num_shards=4, index=0)
    test_data = load_dataset('imdb', split='test')
    test_data = data.shard(num_shards=4, index=0)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # now tokenizer
    tokenized_training_data = tokenizer(train_data['text'], return_tensors='pt', padding=True, truncation=True, max_length=64)
    tokenized_testing_data = tokenizer(test_data['text'], return_tensors='pt', padding=True, truncation=True, max_length=64)
    print(tokenized_testing_data)


    # tokeinzing row by row
    def tokenize_function(text_data):
        return tokenizer(text_data['text'], return_tensors='pt', padding=True, truncation=True, max_length=64)
    # Tokenize in batch size
    tokenized_in_batches = train_data.map(tokenize_function, batched=True)
    tokenized_by_row = train_data.map(tokenize_function, batched=False)



    # Training Arguments
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir="./fintuned",
        evaluation_strategy='epocy',
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        wieight_decay=0.01
    )


    # Trainer Class
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(...)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_data,
        eval_dataset = tokenized_test_data,
        tokenizer=tokenizer
    )


    # Using the fine-tuned model
    new_data = ['this is movie']
    new_input = tokenizer(new_data, return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**new_input)
    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    for i, predicted_label in enumerate(predicted_labels):
        sentiment = label_map[predicted_label]
        print(f'\nInput Text {i + 1}: {new_data[i]}')
        print(f'Predicted Label: {sentiment}')


    # Save Models
    model.save_pretrained('my_fintuned_files')
    tokenizer.save_pretrained('my_fintuned_files')
    model = AutoModelForSequenceClassification.from_pretrained('my_finetuned_files')
    tokenizer = AutoTokenizer.from_pretrained('my_finetuned_files')

































