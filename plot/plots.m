% Default
load('CartPole-v0_1.mat')
figure(1)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('CartPole-v0')

% Cart Mass = 5
load('CartPole-v0_2.mat')
figure(2)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('CartPole-v0')

% Pole Mass = 0.5
load('CartPole-v0_3.mat')
figure(3)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('CartPole-v0')

% Pole Length = 2.5
load('CartPole-v0_4.mat')
figure(4)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('CartPole-v0')

% Default
load('Acrobot-v1_1.mat')
figure(5)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Acrobot-v1')

% LINK_LENGTH_1 = 2, LINK_COM_POS_1 = 1
load('Acrobot-v1_2.mat')
figure(6)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Acrobot-v1')

% LINK_LENGTH_2 = 2, LINK_COM_POS_2 = 1
load('Acrobot-v1_3.mat')
figure(7)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Acrobot-v1')

% LINK_MASS_1 = 2
load('Acrobot-v1_4.mat')
figure(8)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Acrobot-v1')

% LINK_MASS_2 = 2
load('Acrobot-v1_5.mat')
figure(9)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Acrobot-v1')

% Default
load('Pendulum-v0_1.mat')
figure(10)
plot(episode_reward)
xlabel('episode')
ylabel('reward')
title('Pendulum-v0')