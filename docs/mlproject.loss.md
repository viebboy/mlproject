# mlproject.loss Module

This module contains some popular loss functions and convenient interfaces to compose loss functions.

Basically, in supervised learning, a loss function takes 2 inputs: the predictions from the model and the groundtruth labels. 

A loss function returns a single value that reflects the cost of the optimization problem (we always minize a loss function)

We have the following standard loss functions:

- `mlproject.loss.CrossEntropy`: cross entropy loss function in classification
- `mlproject.loss.MSE`: mean-squared error loss function
- `mlproject.loss.MAE`: mean-absolute error loss function
- `mlproject.loss.CosineDissimilarity`: Cosine dissimilarity loss function, basically 1 - cosine.

In addition, we have the following convenient interface to compose loss functions:

## mlproject.loss.compose_losses_to_callable

Give a (nested) list of loss functions, we could use `compose_losses_to_callable()` to combine these losses and generate a single loss function.

For example, our model predicts both the stock movements (classification problem) and the future prices (regression problem).

That is, `model_predictions = model(inputs)` where `model_predictions = (predicted_movements, predicted_prices)`.

To train the model, we could use cross-entropy loss to optimize for predicted movements and mean-squared error loss to optimize for predicted prices.

Basically, we could use implement as loss function by ourselves as follows:

```python
def loss_function(model_predictions, groundtruths):
    predicted_movements, predicted_prices = model_predictions
    true_movements, true_prices = groundtruths
    movement_cost = CrossEntropy(predicted_movements, true_movements)
    price_cost = MSE(predicted_prices, true_prices)
    return movement_cost + price_cost
```

Instead of writing the above lines of code, we could simply do `combined_loss = compose_losses_to_callable([CrossEntropy, MSE])`

`combined_loss` will be just the same as the above `loss_function`.

The above example is simple since the model's outputs is only a 2-element list. 

In practice, `mlproject.loss.compose_losses_to_callable()` works with any nested structures.

You only need to wrap the loss callables in the same nested structure as the model's outputs. 

For example, if the model's outputs has this structure `(prediction1, (prediction2, prediction3), ((prediction4, prediction5), prediction6))`,

then we should pass the nested loss callables in the same structure as `compose_losses_to_callable((loss1, (loss2, loss3), ((loss4, loss5), loss6))`.

Furthermore, we could also weigh the contribution of each loss component to the combined loss function via the second argument to `compose_losses_to_callable()`.

For example, `combined_loss = compose_losses_to_callable([CrossEntropy, MSE], [2, 5])` means multiplying 2 to the cross-entropy loss result and multiply 5 to the mean-squared error loss before summing them up.
