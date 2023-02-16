
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model import SiameseNetwork, ContrastiveLoss
from dataset import ImagePairDataset


def main():
    epochs = 1000
    siamese_net = SiameseNetwork()
    loss_function = ContrastiveLoss()

    pair_data = pd.read_csv("csv/pairs.csv")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    dataset = ImagePairDataset(pair_data, transform)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.SGD(siamese_net.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter()
    for epoch in range(epochs):
        print("started epoch: ", epoch)
        running_loss = 0.0
        for i, (image1, image2, label) in enumerate(train_loader):

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass

            similarity = siamese_net(image1, image2)
            loss = loss_function(similarity, label)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                last_loss = running_loss / 20
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, last_loss))
                writer.add_scalar('Loss/train', last_loss, epoch *len(train_loader) +i + 1)
                running_loss = 0.0

        writer.flush()
    print('Finished Training')


if __name__ == '__main__':
    main()

