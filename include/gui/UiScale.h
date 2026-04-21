#pragma once

#include <QFontMetrics>
#include <QLatin1Char>
#include <QWidget>
#include <QtGlobal>

namespace kfusion {
namespace gui {

/** Average Latin letter width for layout (avoids zero from monospace edge cases). */
inline int uiCharAdvance(const QWidget& w) {
    const QFontMetrics fm(w.font());
    const int n = fm.horizontalAdvance(QLatin1Char('n'));
    const int M = fm.horizontalAdvance(QLatin1Char('M'));
    return qMax(1, qMax(n, M * 3 / 4));
}

/** Side panel widths scale with the widget font (HiDPI, accessibility, distro font sizes). */
inline void setPanelWidthInChars(QWidget& w, int minChars, int maxChars) {
    const int cw = uiCharAdvance(w);
    w.setMinimumWidth(qMax(1, minChars * cw));
    w.setMaximumWidth(qMax(w.minimumWidth(), maxChars * cw));
}

/** Thin progress bars that still read clearly at larger fonts. */
inline int uiThinBarHeight(const QWidget& w) {
    const int h = QFontMetrics(w.font()).height();
    return qMax(6, h / 3);
}

} // namespace gui
} // namespace kfusion
