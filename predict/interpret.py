from predict.tft_inference import model, val_loader
import matplotlib.pyplot as plt

# Feature-Importances anzeigen
interpretation = model.interpret_output(val_loader, reduction="sum")
interpretation.plot()
plt.savefig("outputs/plots/feature_importance.png")
