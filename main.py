import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import EarlyStopping, load_data
import argparse
from utils import setup
import warnings
from model import HoGANet
warnings.filterwarnings("ignore")

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    accuracy = accuracy_score(labels, prediction)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    f1 = f1_score(labels, prediction)
    return accuracy, f1, precision, recall


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, f1, precision, recall = score(logits[mask], labels[mask])

    return loss, accuracy, f1, precision, recall


def main(args):
    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args["dataset"])

    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    model = HoGANet(
        num_meta_paths=len(g),
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    ).to(args["device"])
    g = [graph.to(args["device"]) for graph in g]
    print(model)

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    for epoch in range(args["num_epochs"]):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_f1, train_pre, train_rec = score(
            logits[train_mask], labels[train_mask]
        )
        val_loss, val_acc, val_f1, val_pre, val_rec = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        if epoch%10 == 0:
            print(
                "Epoch {:d} | Train Loss {:.4f} | Train f1 {:.4f} | "
                "Val Loss {:.4f} | Val f1 {:.4f}".format(
                    epoch + 1,
                    loss.item(),
                    train_f1,
                    val_loss.item(),
                    val_f1,
                )
            )
        
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_f1, test_pre, test_rec = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )

    print("Acc For Test:",round(test_acc,4))
    print("Pre For Test:",round(test_pre,4))
    print("Rec For Test:",round(test_rec,4))
    print("F1 For Test:",round(test_f1,4))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("HoGANet")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    args = parser.parse_args().__dict__

    args = setup(args)
    print("Running:", args["device"])
    main(args)