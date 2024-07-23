from tonic.torch import models, normalizers
import torch

def ppo_mlp_model(
  actor_sizes=(64, 64),
  actor_activation=torch.nn.Tanh,
  critic_sizes=(64, 64),
  critic_activation=torch.nn.Tanh,
):

  """
  Constructs an ActorCritic model with specified architectures for the actor and critic networks.

  Parameters:
  - actor_sizes (tuple): Sizes of the layers in the actor MLP.
  - actor_activation (torch activation): Activation function used in the actor MLP.
  - critic_sizes (tuple): Sizes of the layers in the critic MLP.
  - critic_activation (torch activation): Activation function used in the critic MLP.

  Returns:
  - models.ActorCritic: An ActorCritic model comprising an actor and a critic with MLP torsos,
    equipped with a Gaussian policy head for the actor and a value head for the critic,
    along with observation normalization.
  """

  return models.ActorCritic(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(actor_sizes, actor_activation),
      head=models.DetachedScaleGaussianPolicyHead(),
    ),
    critic=models.Critic(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(critic_sizes, critic_activation),
      head=models.ValueHead(),
    ),
    observation_normalizer=normalizers.MeanStd(),
  )