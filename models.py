import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 2: Define the classifier
class ProteinClassifier(nn.Module):
    def __init__(self, esm_model, hidden_size, num_classes):
        super(ProteinClassifier, self).__init__()
        self.esm_model = esm_model
        self.classifier = nn.Sequential(
            nn.Linear(esm_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs):
        with torch.no_grad():  # Freeze the ESM model's forward pass
            input_ids = inputs["input_ids"].squeeze(1)  
            attention_mask = inputs["attention_mask"].squeeze(1)  
            outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state for the sequence (excluding special tokens)
        last_hidden_states = outputs.last_hidden_state
        
        # Remove [CLS] token embeddings and update attention mask
        hidden_states = last_hidden_states[:, 1:, :]  # Exclude [CLS] token
        # print("attention_mask.shape: ", attention_mask.shape)
        attention_mask = attention_mask[:, 1:]  # Exclude [CLS] token from the mask
        # print("hidden_states.shape: ", hidden_states.shape)
        
        # Expand attention mask to match the hidden size dimension
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())   # [batch_size, seq_len - 1, hidden_size]
        # mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()) 
        # print("mask.shape: ", mask.shape)

        # Apply mask: zero out features for padded tokens
        masked_hidden_states = hidden_states * mask

        # Calculate the sum of features and divide by the number of valid tokens
        sum_features = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_size]
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        pooled_feature = sum_features / valid_token_counts  # [batch_size, hidden_size]

        # print("pooled_feature.shape: ", pooled_feature.shape)
        # print("pooled feature: ", pooled_feature)

        # Pass through the classifier
        logits = self.classifier(pooled_feature)
        # print("logits: ", logits)
        return logits
    

class ProteinClassifier_student(nn.Module):
    def __init__(self, esm_model, hidden_size1, hidden_size2, num_classes):
        super(ProteinClassifier_student, self).__init__()
        self.esm_model = esm_model
        self.classifier = nn.Sequential(
            nn.Linear(esm_model.config.hidden_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes),
        )

    def forward(self, inputs):
        with torch.no_grad():  # Freeze the ESM model's forward pass
            # Flatten inputs into the correct format for the ESM model
            input_ids = inputs["input_ids"].squeeze(1)  # Remove extra batch dimension
            attention_mask = inputs["attention_mask"].squeeze(1)  # Remove extra batch dimension

            # Pass the processed inputs to the model
            outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state for the sequence (excluding special tokens)
        last_hidden_states = outputs.last_hidden_state
        
        # Remove [CLS] token embeddings and update attention mask
        hidden_states = last_hidden_states[:, 1:, :]  # Exclude [CLS] token
        # print("attention_mask.shape: ", attention_mask.shape)
        attention_mask = attention_mask[:, 1:]  # Exclude [CLS] token from the mask
        # print("hidden_states.shape: ", hidden_states.shape)
        

        # Expand attention mask to match the hidden size dimension
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())   # [batch_size, seq_len - 1, hidden_size]
        # mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()) 
        # print("mask.shape: ", mask.shape)

        # Apply mask: zero out features for padded tokens
        masked_hidden_states = hidden_states * mask

        # Calculate the sum of features and divide by the number of valid tokens
        sum_features = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_size]
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        pooled_feature = sum_features / valid_token_counts  # [batch_size, hidden_size]

        # print("pooled_feature.shape: ", pooled_feature.shape)
        # print("pooled feature: ", pooled_feature)

        # Pass through the classifier
        logits = self.classifier(pooled_feature)
        # print("logits: ", logits)
        return logits
    

class ProteinClassifier_v3(nn.Module):
    def __init__(self, esm_model, hidden_size, num_classes):
        super(ProteinClassifier_v3, self).__init__()
        self.esm_model = esm_model
        self.classifier = nn.Sequential(
            nn.Linear(esm_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs):
        # Process inputs through the ESM model
        input_ids = inputs["input_ids"].squeeze(1)  # Remove extra batch dimension
        attention_mask = inputs["attention_mask"].squeeze(1)  # Remove extra batch dimension

        # Pass inputs through the ESM model
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state and exclude [CLS] token
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        hidden_states = last_hidden_states[:, 1:, :]  # Exclude the [CLS] token
        attention_mask = attention_mask[:, 1:]  # Exclude the [CLS] token from the mask

        # Mask padded tokens and compute pooled features
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())  # [batch_size, seq_len - 1, hidden_size]
        masked_hidden_states = hidden_states * mask  # Zero out padded token features

        # Compute the sum of features and normalize by the number of valid tokens
        sum_features = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_size]
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        pooled_feature = sum_features / valid_token_counts  # Normalize features

        # Pass pooled features through the classifier
        logits = self.classifier(pooled_feature)
        return logits
    
    
class T5Classifier(nn.Module):
    def __init__(self, t5_model, hidden_size, num_classes):
        super(T5Classifier, self).__init__()
        self.t5_model = t5_model
        self.classifier = nn.Sequential(
            nn.Linear(t5_model.config.d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs):
        with torch.no_grad():  # Freeze the ESM model's forward pass
            # # Flatten inputs into the correct format for the ESM model
            # input_ids = inputs["input_ids"].squeeze(1)  # Remove extra batch dimension
            # attention_mask = inputs["attention_mask"].squeeze(1)  # Remove extra batch dimension
            input_ids = torch.tensor(inputs['input_ids'])
            attention_mask = torch.tensor(inputs['attention_mask'])

            # Pass the processed inputs to the model
            outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state for the sequence (excluding special tokens)
        last_hidden_states = outputs.last_hidden_state
        
        # Remove the last special token (if any) before padding
        # Identify the actual sequence lengths based on the attention mask
        sequence_lengths = attention_mask.sum(dim=1) - 1  # Subtract 1 to exclude the special token
        batch_size, seq_len, hidden_size = last_hidden_states.size()

        # Create a mask to zero out the last special token
        special_token_mask = torch.ones_like(attention_mask)
        for i, length in enumerate(sequence_lengths):
            special_token_mask[i, length] = 0  # Zero out the special token position

        # Update the attention mask to remove the special token and padding
        updated_attention_mask = attention_mask * special_token_mask

        # Expand updated attention mask to match the hidden size dimension
        mask = updated_attention_mask.unsqueeze(-1).expand(last_hidden_states.size())

        # Apply the mask: zero out features for both the special token and padded tokens
        masked_hidden_states = last_hidden_states * mask

        # Calculate the sum of features and divide by the number of valid tokens
        sum_features = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_size]
        valid_token_counts = updated_attention_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]
        pooled_feature = sum_features / valid_token_counts  # [batch_size, hidden_size]

        # pooled_feature now contains the average of the valid token embeddings

        # print("pooled_feature.shape: ", pooled_feature.shape)
        # print("pooled feature: ", pooled_feature)

        # Pass through the classifier
        logits = self.classifier(pooled_feature)
        # print("logits: ", logits)
        return logits
