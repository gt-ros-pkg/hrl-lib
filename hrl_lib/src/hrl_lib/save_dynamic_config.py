#!/usr/bin/python
import rospy
import dynamic_reconfigure.client
import yaml


def main():
    import sys

    if len(sys.argv) < 3:
        print 'save_dynamic_config <node_name> <file.yaml>'
        exit()

    node_name = sys.argv[1]
    filename = sys.argv[2]

    rospy.init_node('save_dynamic_config', anonymous=True)
    client = dynamic_reconfigure.client.Client(node_name)
    node_config = client.get_configuration()

    f = open(filename, 'w')
    rospy.loginfo('Writing config to ' + str(filename))
    new_config = yaml.dump(node_config, f)
    f.close()
