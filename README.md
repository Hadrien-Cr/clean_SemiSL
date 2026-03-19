## Pseudo-Label (`pseudo_label.py`)

```
for x,y in train_loader:
	l_indices = torch.ne(y, NO_LABEL).nonzero(as_tuple=True)
	u_indices = torch.eq(y, NO_LABEL).nonzero(as_tuple=True)
	x1, x2 = augment(x), augment(x)
	out_logits_x1 = model(x1)

	with torch.no_grad(): out_logits_x2 = model(x2)
	    
	out_probs_x1, out_probs_x2 = F.softmax(out_logits_x1, dim = -1), F.softmax(out_logits_x2, dim = -1) 
	    
	supervision_loss = bce_loss(
		out_probs_x1[l_indices],
		F.one_hot(y[l_indices], task["num_classes"]).float()
	)

	pseudo_label = rescale_probs(out_probs_x2, hyperparams.sharpening_temperature)
        
	regularization_loss = kl_div_loss(
	        out_probs_x1,
		pseudo_label
	)

	loss = supervision_loss + regularization_coeff * regularization_loss

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```  


## EntMin (`entmin.py`)

```
for x,y in train_loader:
	l_indices = torch.ne(y, NO_LABEL).nonzero(as_tuple=True)
	u_indices = torch.eq(y, NO_LABEL).nonzero(as_tuple=True)
            
	x = augment(x)
	out_logits = model(x)
	out_probs = F.softmax(out_logits, dim = -1)
            
	supervision_loss = bce_loss(
		out_probs[l_indices],
		F.one_hot(y[l_indices], task["num_classes"]).float()
	)
	
	regularization_loss = entropy(out_probs)

	loss = supervision_loss + regularization_coeff * regularization_loss

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```  

