/****************************************************************************
** Meta object code from reading C++ file 'ControlPanel.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.18)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../include/gui/ControlPanel.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ControlPanel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.18. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_kfusion__gui__ControlPanel_t {
    QByteArrayData data[22];
    char stringdata0[291];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_kfusion__gui__ControlPanel_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_kfusion__gui__ControlPanel_t qt_meta_stringdata_kfusion__gui__ControlPanel = {
    {
QT_MOC_LITERAL(0, 0, 26), // "kfusion::gui::ControlPanel"
QT_MOC_LITERAL(1, 27, 12), // "startClicked"
QT_MOC_LITERAL(2, 40, 0), // ""
QT_MOC_LITERAL(3, 41, 11), // "stopClicked"
QT_MOC_LITERAL(4, 53, 12), // "resetClicked"
QT_MOC_LITERAL(5, 66, 16), // "exportPLYClicked"
QT_MOC_LITERAL(6, 83, 16), // "exportGLBClicked"
QT_MOC_LITERAL(7, 100, 11), // "modeChanged"
QT_MOC_LITERAL(8, 112, 5), // "index"
QT_MOC_LITERAL(9, 118, 14), // "threadsChanged"
QT_MOC_LITERAL(10, 133, 1), // "n"
QT_MOC_LITERAL(11, 135, 23), // "hyperparamsApplyClicked"
QT_MOC_LITERAL(12, 159, 21), // "cameraRotationChanged"
QT_MOC_LITERAL(13, 181, 5), // "pitch"
QT_MOC_LITERAL(14, 187, 3), // "yaw"
QT_MOC_LITERAL(15, 191, 4), // "roll"
QT_MOC_LITERAL(16, 196, 17), // "onPipelineStarted"
QT_MOC_LITERAL(17, 214, 17), // "onPipelineStopped"
QT_MOC_LITERAL(18, 232, 16), // "setExportEnabled"
QT_MOC_LITERAL(19, 249, 7), // "enabled"
QT_MOC_LITERAL(20, 257, 17), // "setCameraRotation"
QT_MOC_LITERAL(21, 275, 15) // "onPresetChanged"

    },
    "kfusion::gui::ControlPanel\0startClicked\0"
    "\0stopClicked\0resetClicked\0exportPLYClicked\0"
    "exportGLBClicked\0modeChanged\0index\0"
    "threadsChanged\0n\0hyperparamsApplyClicked\0"
    "cameraRotationChanged\0pitch\0yaw\0roll\0"
    "onPipelineStarted\0onPipelineStopped\0"
    "setExportEnabled\0enabled\0setCameraRotation\0"
    "onPresetChanged"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_kfusion__gui__ControlPanel[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   84,    2, 0x06 /* Public */,
       3,    0,   85,    2, 0x06 /* Public */,
       4,    0,   86,    2, 0x06 /* Public */,
       5,    0,   87,    2, 0x06 /* Public */,
       6,    0,   88,    2, 0x06 /* Public */,
       7,    1,   89,    2, 0x06 /* Public */,
       9,    1,   92,    2, 0x06 /* Public */,
      11,    0,   95,    2, 0x06 /* Public */,
      12,    3,   96,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      16,    0,  103,    2, 0x0a /* Public */,
      17,    0,  104,    2, 0x0a /* Public */,
      18,    1,  105,    2, 0x0a /* Public */,
      20,    3,  108,    2, 0x0a /* Public */,
      21,    1,  115,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int,   13,   14,   15,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int,   13,   14,   15,
    QMetaType::Void, QMetaType::Int,    8,

       0        // eod
};

void kfusion::gui::ControlPanel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ControlPanel *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->startClicked(); break;
        case 1: _t->stopClicked(); break;
        case 2: _t->resetClicked(); break;
        case 3: _t->exportPLYClicked(); break;
        case 4: _t->exportGLBClicked(); break;
        case 5: _t->modeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->threadsChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->hyperparamsApplyClicked(); break;
        case 8: _t->cameraRotationChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 9: _t->onPipelineStarted(); break;
        case 10: _t->onPipelineStopped(); break;
        case 11: _t->setExportEnabled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: _t->setCameraRotation((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 13: _t->onPresetChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::startClicked)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::stopClicked)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::resetClicked)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::exportPLYClicked)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::exportGLBClicked)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::modeChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::threadsChanged)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::hyperparamsApplyClicked)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (ControlPanel::*)(int , int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel::cameraRotationChanged)) {
                *result = 8;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject kfusion::gui::ControlPanel::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_kfusion__gui__ControlPanel.data,
    qt_meta_data_kfusion__gui__ControlPanel,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *kfusion::gui::ControlPanel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *kfusion::gui::ControlPanel::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_kfusion__gui__ControlPanel.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int kfusion::gui::ControlPanel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 14;
    }
    return _id;
}

// SIGNAL 0
void kfusion::gui::ControlPanel::startClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void kfusion::gui::ControlPanel::stopClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void kfusion::gui::ControlPanel::resetClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void kfusion::gui::ControlPanel::exportPLYClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void kfusion::gui::ControlPanel::exportGLBClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void kfusion::gui::ControlPanel::modeChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void kfusion::gui::ControlPanel::threadsChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void kfusion::gui::ControlPanel::hyperparamsApplyClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 7, nullptr);
}

// SIGNAL 8
void kfusion::gui::ControlPanel::cameraRotationChanged(int _t1, int _t2, int _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
