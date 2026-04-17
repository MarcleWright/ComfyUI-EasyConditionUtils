import { app } from "../../../scripts/app.js";

const MAX_TEXT_SLOTS = 50;

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

function cacheOriginalWidgets(node) {
  if (node._easyTextWidgetMap) {
    return;
  }

  const widgetMap = new Map();
  for (const widget of node.widgets || []) {
    if (!widget?.name) continue;
    widgetMap.set(widget.name, widget);
  }
  node._easyTextWidgetMap = widgetMap;
}

function getCachedWidget(node, name) {
  return node._easyTextWidgetMap?.get(name) || null;
}

function clearHiddenSlots(node, visibleCount) {
  for (let slotIndex = visibleCount + 1; slotIndex <= MAX_TEXT_SLOTS; slotIndex += 1) {
    const widget = getCachedWidget(node, `text_${String(slotIndex).padStart(2, "0")}`);
    if (widget) {
      widget.value = "";
    }
  }
}

function rebuildWidgetOrder(node) {
  cacheOriginalWidgets(node);

  const countWidget = getCachedWidget(node, "count");
  const indexWidget = getCachedWidget(node, "index");
  if (countWidget && (countWidget.value == null || Number.isNaN(Number(countWidget.value)))) {
    countWidget.value = 1;
  }
  if (indexWidget && (indexWidget.value == null || Number.isNaN(Number(indexWidget.value)))) {
    indexWidget.value = 0;
  }

  const rawCount = Number(countWidget?.value ?? 1);
  const visibleCount = Math.max(1, Math.min(MAX_TEXT_SLOTS, Number.isFinite(rawCount) ? rawCount : 1));
  if (countWidget) {
    countWidget.value = visibleCount;
  }

  clearHiddenSlots(node, visibleCount);

  const ordered = [];
  for (let slotIndex = 1; slotIndex <= visibleCount; slotIndex += 1) {
    const widget = getCachedWidget(node, `text_${String(slotIndex).padStart(2, "0")}`);
    if (widget) {
      ordered.push(widget);
    }
    if (slotIndex < visibleCount) {
      ordered.push(makeSpacer(`text_slot_spacer_${slotIndex}`, 8));
    }
  }

  ordered.push(makeSpacer("text_controls_spacer", 18));
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
      const rememberedWidth = Number.isFinite(node._easyTextRememberedWidth) && node._easyTextRememberedWidth > 0
        ? node._easyTextRememberedWidth
        : (Array.isArray(node.size) && Number.isFinite(node.size[0]) && node.size[0] > 0 ? node.size[0] : computed[0]);
      node._easyTextRememberedWidth = rememberedWidth;
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
      this._easyTextRememberedWidth = width;
    }
    return result;
  };

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
    if (Array.isArray(this.size) && Number.isFinite(this.size[0]) && this.size[0] > 0) {
      this._easyTextRememberedWidth = this.size[0];
    }
    refreshNodeLayout(this);
    return result;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
    if (Array.isArray(this.size) && Number.isFinite(this.size[0]) && this.size[0] > 0) {
      this._easyTextRememberedWidth = this.size[0];
    }
    refreshNodeLayout(this);
    return result;
  };
}

app.registerExtension({
  name: "EasyConditionUtils.EasyTextListSelectorWidgets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "EasyTextListSelector") {
      return;
    }

    hookCountReactivity(nodeType);
    hookLifecycle(nodeType);
  },
});
