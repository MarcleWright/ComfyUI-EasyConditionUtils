import { app } from "../../../scripts/app.js";

function isDynamicImageName(name) {
  return /^img_\d{2}$/.test(name);
}

function inputIndex(name) {
  const match = name.match(/^img_(\d{2})$/);
  return match ? Number(match[1]) : null;
}

function nextImageName(index) {
  return `img_${String(index).padStart(2, "0")}`;
}

function getDynamicInputs(node) {
  return (node.inputs || [])
    .map((input, slot) => ({ input, slot, index: inputIndex(input.name) }))
    .filter((item) => item.index != null)
    .sort((a, b) => a.index - b.index);
}

function syncDynamicImageInputs(node) {
  const dynamicInputs = getDynamicInputs(node);
  if (!dynamicInputs.length) {
    node.addInput("img_01", "IMAGE");
    return;
  }

  let highestConnected = 0;
  for (const item of dynamicInputs) {
    if (item.input.link != null) {
      highestConnected = Math.max(highestConnected, item.index);
    }
  }

  const keepUntil = Math.max(1, highestConnected + 1);

  for (let i = node.inputs.length - 1; i >= 0; i -= 1) {
    const input = node.inputs[i];
    const index = inputIndex(input.name);
    if (index == null) continue;
    if (index > keepUntil) {
      node.removeInput(i);
    }
  }

  const existingNames = new Set((node.inputs || []).map((input) => input.name));
  for (let index = 1; index <= keepUntil; index += 1) {
    const name = nextImageName(index);
    if (!existingNames.has(name)) {
      node.addInput(name, "IMAGE");
    }
  }
}

app.registerExtension({
  name: "EasyConditionUtils.EasyFlux2KleinConditionDynamicInputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "EasyFlux2KleinCondition") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      syncDynamicImageInputs(this);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
      syncDynamicImageInputs(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      syncDynamicImageInputs(this);
      return result;
    };
  },
});
