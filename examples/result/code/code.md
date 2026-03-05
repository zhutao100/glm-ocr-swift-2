relationship between the beans. The only difference from previous exercises is the change in the JNDI name element tag for the Address home interface:

```xml

||| <local-jndi-name>AddressHomeLocal</local-jndi-name>

```

Because the Home interface for the Address is local, the tag is `<local-jndi-name>` rather than `<jndi-name>`.

The **weblogic-emp-rdbms-jar.xml** descriptor file contains a number of new sections and elements in this exercise. A detailed examination of the relationship elements will await the next section.

The file contains a section mapping the `Address` `<camp-field>` attributes from the `ebi-jar.xml` file to the database columns in the `ANIMADEX` table to a new section related to the database. The key values in this section are the key values in this file.

```html

<weblogic-rdmms-bean>

    <ejb-name>AddressJUE</ejb-name>

    <data-source-name>itran-dataSource</data-source-name>

    <table-name>ADDRESS</table-name>

    <field-map>

        <cmp-field>id</cmp-field>

        <dbms-column>ID</dbms-column>

    </field-map>

    <field-map>

        <cmp-field>street</cmp-field>

        <dbms-column>STREET</dbms-column>

    </field-map>

    <field-map>

        <cmp-field>city</cmp-field>

        <dbms-column>CITY</dbms-column>

    </field-map>

    <field-map>

        <cmp-field>state</cmp-field>

        <dbms-column>STATE</dbms-column>

    </field-map>

    <field-map>

        <cmp-field>zip</cmp-field>

        <dbms-column>ZIP</dbms-column>

    </field-map>

    <!-- automatically generate the value of ID in the database on

inserts using sequence table -->

    <automatic-key-generation>

        <generator-type>NAMED_SEQUENCE_TABLE</generator-type>

        <generator-name>ADDRESS_SEQUENCE</generator-name>

        <key-cache-size>/Key-cache-size>

    </automatic-key-generation>

</weblogic-rdmms-bean>

```
