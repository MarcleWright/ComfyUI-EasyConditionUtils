import { app } from "../../../scripts/app.js";

function inputIndex(name) {
  const match = name.match(/^img_(\d{2})$/);
  return match ? Number(match[1]) : null;
}

function nextImageName(index) {
  return `img_${String(index).padStart(2, "0")}`;
}

function nextWeightName(index) {
  return `${nextImageName(index)}_weight`;
}

function getDynamicImageInputs(node) {
  return (node.inputs || [])
    .map((input, slot) => ({ input, slot, index: inputIndex(input.name) }))
    .filter((item) => item.index != null)
    .sort((a, b) => a.index - b.index);
}

function getWidgetIndex(node, name) {
  return (node.widgets || []).findIndex((widget) => widget.name === name);
}

function removeWidgetByName(node, name) {
  const widgetIndex = getWidgetIndex(node, name);
  if (widgetIndex === -1) {
    return;
  }

  node.widgets.splice(widgetIndex, 1);
  if (Array.isArray(node.widgets_values) && widgetIndex < node.widgets_values.length) {
    node.widgets_values.splice(widgetIndex, 1);
  }
}

function ensureWeightWidgets(node, keepUntil) {
  for (let index = 1; index <= keepUntil; index += 1) {
    const name = nextWeightName(index);
    const existingIndex = getWidgetIndex(node, name);
    if (existingIndex !== -1) {
      const existingWidget = node.widgets[existingIndex];
      if (existingWidget.value == null || Number.isNaN(existingWidget.value)) {
        existingWidget.value = 1.0;
      }
      continue;
    }

    const widget = node.addWidget("number", name, 1.0, () => {}, {
      min: 0.0,
      max: 8.0,
      step: 0.05,
      precision: 2,
    });
    widget.value = 1.0;
  }

  for (let index = keepUntil + 1; index < 100; index += 1) {
    const name = nextWeightName(index);
    if (getWidgetIndex(node, name) === -1) {
      break;
    }
    removeWidgetByName(node, name);
  }
}

function syncDynamicImageInputs(node, includeWeightWidgets = false) {
  const dynamicInputs = getDynamicImageInputs(node);
  if (!dynamicInputs.length) {
    node.addInput("img_01", "IMAGE");
    if (includeWeightWidgets) {
      ensureWeightWidgets(node, 1);
    }
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

  if (includeWeightWidgets) {
    ensureWeightWidgets(node, keepUntil);
  }
}

function registerDynamicNode(nodeType, includeWeightWidgets) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
    syncDynamicImageInputs(this, includeWeightWidgets);
    return result;
  };

  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function () {
    const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
    syncDynamicImageInputs(this, includeWeightWidgets);
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
    syncDynamicImageInputs(this, includeWeightWidgets);
    return result;
  };
}

app.registerExtension({
  name: "EasyConditionUtils.EasyFlux2KleinConditionDynamicInputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "EasyFlux2KleinCondition") {
      registerDynamicNode(nodeType, false);
      return;
    }

    if (nodeData.name === "EasyFlux2KleinConditionAdvanced") {
      registerDynamicNode(nodeType, true);
    }
  },
});
