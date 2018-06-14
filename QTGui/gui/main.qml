import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.2
 
ApplicationWindow {
    visible: true
    width: 300
    height: 240
    title: qsTr("PyQt5 love QML")
    color: "whitesmoke"
 
    GridLayout {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 9
 
        columns: 2
        rows: 2
        rowSpacing: 10
        columnSpacing: 10
 
        Text {
            id: visibletext
            text: qsTr("First number")
        }
 
        Button {
            height: 40
            Layout.fillWidth: true
            text: qsTr("Sum numbers")
 
            Layout.columnSpan: 2
 
            onClicked: {
                main_controller.on_click_button()
            }
        }
    }
 
    Connections {
        target: main_controller 
        onClickResult: {
            visibletext.text = text_result
        }
    }
}