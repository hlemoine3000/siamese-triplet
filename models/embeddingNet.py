"""Discriminator model for ADDA."""

from torch import nn


class EmbeddingNet(nn.Module):
    def __init__(self, n_outputs=128):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            #nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.n_classes = 10
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(nn.Linear(50 * 4 * 4, 500),
                                nn.ReLU(),
                                # nn.Linear(512, 256),
                                # nn.ReLU(),
                                nn.Linear(500, self.n_outputs)
                                )

    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)