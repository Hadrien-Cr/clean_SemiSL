## PI-Model (`pi_model.py`)
```
    for iteration in range(hyperparams.num_iterations):
        model.train()
        
        (x_u, _), (x_l,y_l) = next(train_loader)
        n_u = len(x_u)
        x = torch.cat([x_u, x_l], dim=0)
        x = x.to(cfg.device)
        y_l = y_l.to(cfg.device)

        x1, x2 = augment(x), augment(x)
        out_logits_x1 = model(x1)
         
        with torch.no_grad(): out_logits_x2 = model(x2)
            
        out_probs_x1, out_probs_x2 = F.softmax(out_logits_x1, dim = -1), F.softmax(out_logits_x2, dim = -1) 
            
        supervision_loss = ce_loss(out_logits_x1[n_u:],y_l)

        regularization_loss = mse_loss(out_probs_x1,out_probs_x2)
    
        loss = supervision_loss + regularization_coeff * regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Entropy Minimization (`entmin.py`)
```
    for iteration in range(hyperparams.num_iterations):
        model.train()
        
        (x_u, _), (x_l,y_l) = next(train_loader)
        n_u = len(x_u)
        x = torch.cat([x_u, x_l], dim=0)
        x = x.to(cfg.device)
        y_l = y_l.to(cfg.device)

        x = augment(x)
        out_logits = model(x)
         
        out_probs = F.softmax(out_logits, dim = -1) 
            
        supervision_loss = ce_loss(out_logits[n_u:],y_l)

        regularization_loss = entropy(out_probs)
    
        loss = supervision_loss + regularization_coeff * regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## RoadMap

- [x] Pi-Model
- [x] Entmin
- [ ] Self-Training
- [ ] Pseudo-Labeling
- [ ] Virtual Adversarial Training
- [ ] Temporal Ensembling
- [ ] Mean Teacher
- [ ] SimCLR
- [ ] MixMatch
- [ ] FixMatch
- [ ] SimMatch
- [ ] Consistency-based for Object Detection
- [ ] Unbiased Teacher for Object Detection

