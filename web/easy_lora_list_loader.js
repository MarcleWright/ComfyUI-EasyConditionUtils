import { app } from "../../../scripts/app.js";

const MAX_LORA_SLOTS = 50;
const LORA_NONE = "None";

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name) || null;
}

function getSlotWidgets(node, slotIndex) {
  const suffix = String(slotIndex).padStart(2, "0");
  const widgetMap = node._easyLoraWidgetMap || new Map();
  return {
    lora: widgetMap.get(`lora_${suffix}`) || null,
    strength: widgetMap.get(`strength_model_${suffix}`) || null,
  };
}

function makeSpacer(name, height) {
  return {
    name,
    type: "easy_spacer",
    value: null,
    draw: () => {},
    computeSize: (width) => [width || 0, height],
    serializeValue: () => undefined,
  };
}

function normalizeStrengthWidget(widget) {
  if (!widget) return;
  if (widget.value == null || Number.isNaN(widget.value)) {
    widget.value = 1.0;
  }
}

function clearHiddenSlots(node, visibleCount) {
  for (let slotIndex = visibleCount + 1; slotIndex <= MAX_LORA_SLOTS; slotIndex += 1) {
    const { lora, strength } = getSlotWidgets(node, slotIndex);
    if (lora) {
      lora.value = LORA_NONE;
    }
    if (strength) {
      strength.value = 1.0;
    }
  }
}

function cacheOriginalWidgets(node) {
  if (node._easyLoraWidgetMap) {
    return;
  }

  const widgetMap = new Map();
  for (const widget of node.widgets || []) {
    if (!widget?.name) continue;
    widgetMap.set(widget.name, widget);
  }
  node._easyLoraWidgetMap = widgetMap;
}

function rebuildWidgetOrder(node) {
  cacheOriginalWidgets(node);

  const countWidget = node._easyLoraWidgetMap.get("count") || null;
  const indexWidget = node._easyLoraWidgetMap.get("index") || null;
  if (countWidget && (countWidget.value == null || Number.isNaN(Number(countWidget.value)))) {
    countWidget.value = 1;
  }
  if (indexWidget && (indexWidget.value == null || Number.isNaN(Number(indexWidget.value)))) {
    indexWidget.value = 0;
  }
  const rawCount = Number(countWidget?.value ?? 1);
  const visibleCount = Math.max(1, Math.min(MAX_LORA_SLOTS, Number.isFinite(rawCount) ? rawCount : 1));
  if (countWidget) {
    countWidget.value = visibleCount;
  }

  clearHiddenSlots(node, visibleCount);

  const ordered = [];
  for (let slotIndex = 1; slotIndex <= visibleCount; slotIndex += 1) {
    const { lora, strength } = getSlotWidgets(node, slotIndex);
    if (lora) {
      ordered.push(lora);
    }
    if (strength) {
      normalizeStrengthWidget(strength);
      ordered.push(strength);
    }
    if (slotIndex < visibleCount) {
      ordered.push(makeSpacer(`slot_spacer_${slotIndex}`, 8));
    }
  }

  ordered.push(makeSpacer("controls_spacer", 18));
  if (countWidget) {
    ordered.push(countWidget);
  }
  if (indexWidget) {
    ordered.push(indexWidget);
  }

  node.widgets = ordered;
}

function refreshNodeLayout(node) {
  rebuildWidgetOrder(node);

  if (typeof node.computeSize === "function") {
    const computed = node.computeSize();
    if (Array.isArray(computed) && computed.length === 2) {
      const rememberedWidth = Number.isFinite(node._easyLoraRememberedWidth) && node._easyLoraRememberedWidth > 0
        ? node._easyLoraRememberedWidth
        : (Array.isArray(node.size) && Number.isFinite(node.size[0]) && node.size[0] > 0 ? node.size[0] : computed[0]);
      node._easyLoraRememberedWidth = rememberedWidth;
      node.setSize?.([rememberedWidth, computed[1]]);
    }
  }

  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function hookCountReactivity(nodeType) {
  const onWidgetChanged = nodeType.prototype.onWidgetChanged;
  nodeType.prototype.onWidgetChanged = function (name, value, oldValue, widget) {
    const result = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined;
    const widgetName = widget?.name || name;
    if (widgetName === "count") {
      refreshNodeLayout(this);
    }
    return result;
  };
}

function hookLifecycle(nodeType) {
  const onResize = nodeType.prototype.onResize;
  nodeType.prototype.onResize = function (size) {
    const result = onResize ? onResize.apply(this, arguments) : undefined;
    const width = Array.isArray(size) ? size[0] : this?.size?.[0];
    if (Number.isFinite(width) && width > 0) {
      this._easyLoraRememberedWidth = width;
    }
    return result;
  };

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
    if (Array.isArray(this.size) && Number.isFinite(this.size[0]) && this.size[0] > 0) {
      this._easyLoraRememberedWidth = this.size[0];
    }
    refreshNodeLayout(this);
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
    if (Array.isArray(this.size) && Number.isFinite(this.size[0]) && this.size[0] > 0) {
      this._easyLoraRememberedWidth = this.size[0];
    }
    refreshNodeLayout(this);
    return result;
  };
}

app.registerExtension({
  name: "EasyConditionUtils.EasyLoraListLoaderWidgets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "EasyLoraListLoader") {
      return;
    }

    hookCountReactivity(nodeType);
    hookLifecycle(nodeType);
  },
});
